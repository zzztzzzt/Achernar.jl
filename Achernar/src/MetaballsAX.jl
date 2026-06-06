module MetaballsAX

import Axis as AX
using StaticArrays

#=
Public API
=#
export FRAME_INTERVAL
export FIELD_BUFFER, VERTEX_BUFFER, NORMAL_BUFFER, PAYLOAD_BUFFER
export update_physics!, compute_field!, build_mesh!, build_payload!
export init!

#=
Constants
=#
const BALL_COUNT      = 12
const FRAME_INTERVAL  = 1 / 30
const SPEED_LIMIT     = 0.008f0

const GRID_RESOLUTION = 108
const GRID_SIZE       = GRID_RESOLUTION * GRID_RESOLUTION * GRID_RESOLUTION
const CUBE_COUNT      = (GRID_RESOLUTION - 1) * (GRID_RESOLUTION - 1) * (GRID_RESOLUTION - 1)
const MAX_TRIANGLES   = CUBE_COUNT * 5
const ISOLEVEL        = 80.0f0
const SUBTRACT        = 8.0f0
const FIELD_EPSILON   = 1.0f-6

#=
GPU Resource IDs
=#
const _BUF_FIELD      = 10
const _BUF_GRID_AXIS  = 11
const _BUF_BLOB_DATA  = 12
const _BUF_PARAMS     = 13
const _PIPELINE_ID    = 10
const _WORKGROUP_SIZE = (8, 8, 4)

#=
SoA Pre-allocated Storage
=#
const BLOB_X    = Vector{Float32}(undef, BALL_COUNT)
const BLOB_Y    = Vector{Float32}(undef, BALL_COUNT)
const BLOB_Z    = Vector{Float32}(undef, BALL_COUNT)
const BLOB_VX   = Vector{Float32}(undef, BALL_COUNT)
const BLOB_VY   = Vector{Float32}(undef, BALL_COUNT)
const BLOB_VZ   = Vector{Float32}(undef, BALL_COUNT)
const BLOB_SIZE = Vector{Float32}(undef, BALL_COUNT)

const GRID_AXIS = Float32[(i - 1) / Float32(GRID_RESOLUTION - 1) for i in 1:GRID_RESOLUTION]

# Reusable Buffers
const FIELD_BUFFER   = Vector{Float32}(undef, GRID_SIZE)
const VERTEX_BUFFER  = Vector{Float32}(undef, MAX_TRIANGLES * 9)
const NORMAL_BUFFER  = Vector{Float32}(undef, MAX_TRIANGLES * 9)
const PAYLOAD_BUFFER = Vector{Float32}(undef, 1 + MAX_TRIANGLES * 18)
const _BLOB_DATA_BUF = Vector{Float32}(undef, BALL_COUNT * 4)
const _PARAMS_BUF    = Vector{UInt8}(undef, 16)

include("utils/MarchingCubesTables.jl")

@AX.rust_code """
static mut EDGE_TABLE: [i32; 256] = [0; 256];
static mut TRI_TABLE: [i32; 4096] = [0; 4096];

static EDGE_VERTEX_INDICES: [(usize, usize); 12] = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7)
];

#[inline(always)]
fn rust_sample_gradient(x: f32, y: f32, z: f32, bx: &[f32], by: &[f32], bz: &[f32], bsize: &[f32], bc: usize, eps: f32, sub: f32) -> (f32, f32, f32) {
    let mut gx = 0.0; let mut gy = 0.0; let mut gz = 0.0;
    for i in 0..bc {
        let dx = x - bx[i]; let dy = y - by[i]; let dz = z - bz[i];
        let dist_sq = eps + dx*dx + dy*dy + dz*dz;
        let contrib = bsize[i] / dist_sq - sub;
        if contrib > 0.0 {
            let scale = -2.0 * bsize[i] / (dist_sq * dist_sq);
            gx += dx * scale; gy += dy * scale; gz += dz * scale;
        }
    }
    let len = (gx*gx + gy*gy + gz*gz).sqrt();
    if len > 1e-6 { (gx / len, gy / len, gz / len) } else { (0.0, 1.0, 0.0) }
}

#[inline(always)]
fn rust_interpolate(ax: f32, ay: f32, az: f32, av: f32, bx: f32, by: f32, bz: f32, bv: f32, iso: f32) -> (f32, f32, f32) {
    if (bv - av).abs() < 1e-6 { return (ax, ay, az); }
    let t = ((iso - av) / (bv - av)).clamp(0.0, 1.0);
    (ax + (bx - ax)*t, ay + (by - ay)*t, az + (bz - az)*t)
}
"""

@AX.rust_fn function _rust_init_tables!(edge_ptr::Ptr{Int32}, tri_ptr::Ptr{Int32})::Cvoid
    """
    unsafe {
        let edge_dest = std::ptr::addr_of_mut!(EDGE_TABLE) as *mut i32;
        let tri_dest  = std::ptr::addr_of_mut!(TRI_TABLE) as *mut i32;
        
        std::ptr::copy_nonoverlapping(edge_ptr, edge_dest, 256);
        std::ptr::copy_nonoverlapping(tri_ptr, tri_dest, 4096);
    }
    """
end

@AX.rust_fn function _rust_update_physics!(
    bx::Ptr{Float32}, by::Ptr{Float32}, bz::Ptr{Float32},
    bvx::Ptr{Float32}, bvy::Ptr{Float32}, bvz::Ptr{Float32},
    _bsize::Ptr{Float32}, bc::Int32, is_gravity::Int32, speed_limit::Float32
)::Cvoid
    """
    let bc = bc as usize;
    let bx = unsafe { std::slice::from_raw_parts_mut(bx, bc) };
    let by = unsafe { std::slice::from_raw_parts_mut(by, bc) };
    let bz = unsafe { std::slice::from_raw_parts_mut(bz, bc) };
    let bvx = unsafe { std::slice::from_raw_parts_mut(bvx, bc) };
    let bvy = unsafe { std::slice::from_raw_parts_mut(bvy, bc) };
    let bvz = unsafe { std::slice::from_raw_parts_mut(bvz, bc) };
    let is_gravity_on = is_gravity == 1;

    for i in 0..bc {
        bx[i] += bvx[i]; by[i] += bvy[i]; bz[i] += bvz[i];
        for j in 0..bc {
            if i == j { continue; }
            let dx = bx[j] - bx[i]; let dy = by[j] - by[i]; let dz = bz[j] - bz[i];
            let dist_sq = dx*dx + dy*dy + dz*dz + 0.01;
            let force = if is_gravity_on { 0.000008 } else { -0.00002 } / dist_sq;
            bvx[i] += dx * force; bvy[i] += dy * force; bvz[i] += dz * force;
        }
        let margin = 0.15;
        if bx[i] < margin { bvx[i] += 0.001; } else if bx[i] > 1.0 - margin { bvx[i] -= 0.001; }
        if by[i] < margin { bvy[i] += 0.001; } else if by[i] > 1.0 - margin { bvy[i] -= 0.001; }
        if bz[i] < margin { bvz[i] += 0.001; } else if bz[i] > 1.0 - margin { bvz[i] -= 0.001; }
        
        bvx[i] = (bvx[i] * 0.98).clamp(-speed_limit, speed_limit);
        bvy[i] = (bvy[i] * 0.98).clamp(-speed_limit, speed_limit);
        bvz[i] = (bvz[i] * 0.98).clamp(-speed_limit, speed_limit);
    }
    """
end

@AX.rust_fn function _rust_pack_blobs!(
    bx::Ptr{Float32}, by::Ptr{Float32}, bz::Ptr{Float32}, bsize::Ptr{Float32},
    bc::Int32, dest::Ptr{Float32}
)::Cvoid
    """
    let bc = bc as usize;
    let bx = unsafe { std::slice::from_raw_parts(bx, bc) };
    let by = unsafe { std::slice::from_raw_parts(by, bc) };
    let bz = unsafe { std::slice::from_raw_parts(bz, bc) };
    let bsize = unsafe { std::slice::from_raw_parts(bsize, bc) };
    let dest = unsafe { std::slice::from_raw_parts_mut(dest, bc * 4) };
    for i in 0..bc {
        let b = i * 4;
        dest[b]     = bx[i];
        dest[b + 1] = by[i];
        dest[b + 2] = bz[i];
        dest[b + 3] = bsize[i];
    }
    """
end

# Marching Cubes and Analytical Gradient
@AX.rust_fn function _rust_build_mesh!(
    v_out::Ptr{Float32}, n_out::Ptr{Float32}, field::Ptr{Float32}, axis::Ptr{Float32}, res::Int32,
    bx::Ptr{Float32}, by::Ptr{Float32}, bz::Ptr{Float32}, bsize::Ptr{Float32}, bc::Int32,
    isolevel::Float32, sub::Float32, eps::Float32
)::Int32
    """
    let res = res as usize;
    let bc = bc as usize;
    let field = unsafe { std::slice::from_raw_parts(field, res * res * res) };
    let axis = unsafe { std::slice::from_raw_parts(axis, res) };
    let bx = unsafe { std::slice::from_raw_parts(bx, bc) };
    let by = unsafe { std::slice::from_raw_parts(by, bc) };
    let bz = unsafe { std::slice::from_raw_parts(bz, bc) };
    let bsize = unsafe { std::slice::from_raw_parts(bsize, bc) };
    
    let v_out = unsafe { std::slice::from_raw_parts_mut(v_out, res * res * res * 5 * 9) };
    let n_out = unsafe { std::slice::from_raw_parts_mut(n_out, res * res * res * 5 * 9) };
    
    let mut o = 0;
    let mut ex = [0.0f32; 12]; let mut ey = [0.0f32; 12]; let mut ez = [0.0f32; 12];
    let mut enx = [0.0f32; 12]; let mut eny = [0.0f32; 12]; let mut enz = [0.0f32; 12];

    let grid_index = |x: usize, y: usize, z: usize| -> usize { x + y * res + z * res * res };

    unsafe {
        let edges = &*std::ptr::addr_of!(EDGE_TABLE);
        let tris  = &*std::ptr::addr_of!(TRI_TABLE);

        for z in 0..(res - 1) {
            let z0 = axis[z]; let z1 = axis[z+1];
            for y in 0..(res - 1) {
                let y0 = axis[y]; let y1 = axis[y+1];
                for x in 0..(res - 1) {
                    let x0 = axis[x]; let x1 = axis[x+1];
                    
                    let cv = [
                        field[grid_index(x, y, z)],     field[grid_index(x+1, y, z)],
                        field[grid_index(x+1, y+1, z)], field[grid_index(x, y+1, z)],
                        field[grid_index(x, y, z+1)],     field[grid_index(x+1, y, z+1)],
                        field[grid_index(x+1, y+1, z+1)], field[grid_index(x, y+1, z+1)],
                    ];

                    let mut ci = 0;
                    for i in 0..8 { if cv[i] < isolevel { ci |= 1 << i; } }
                    let em = edges[ci];
                    if em == 0 { continue; }

                    let cx = [x0, x1, x1, x0, x0, x1, x1, x0];
                    let cy = [y0, y0, y1, y1, y0, y0, y1, y1];
                    let cz = [z0, z0, z0, z0, z1, z1, z1, z1];

                    for e in 0..12 {
                        if (em & (1 << e)) == 0 { continue; }
                        let (ia, ib) = EDGE_VERTEX_INDICES[e];
                        let (v_x, v_y, v_z) = rust_interpolate(cx[ia], cy[ia], cz[ia], cv[ia], cx[ib], cy[ib], cz[ib], cv[ib], isolevel);
                        ex[e] = v_x; ey[e] = v_y; ez[e] = v_z;
                        
                        let (nx, ny, nz) = rust_sample_gradient(v_x, v_y, v_z, bx, by, bz, bsize, bc, eps, sub);
                        enx[e] = nx; eny[e] = ny; enz[e] = nz;
                    }

                    let mut ti = ci * 16;
                    while tris[ti] != -1 {
                        let e1 = tris[ti] as usize;
                        let e2 = tris[ti+1] as usize;
                        let e3 = tris[ti+2] as usize;

                        v_out[o]   = ex[e1]*2.0 - 1.0; v_out[o+1] = ey[e1]*2.0 - 1.0; v_out[o+2] = ez[e1]*2.0 - 1.0;
                        v_out[o+3] = ex[e2]*2.0 - 1.0; v_out[o+4] = ey[e2]*2.0 - 1.0; v_out[o+5] = ez[e2]*2.0 - 1.0;
                        v_out[o+6] = ex[e3]*2.0 - 1.0; v_out[o+7] = ey[e3]*2.0 - 1.0; v_out[o+8] = ez[e3]*2.0 - 1.0;

                        n_out[o]   = enx[e1]; n_out[o+1] = eny[e1]; n_out[o+2] = enz[e1];
                        n_out[o+3] = enx[e2]; n_out[o+4] = eny[e2]; n_out[o+5] = enz[e2];
                        n_out[o+6] = enx[e3]; n_out[o+7] = eny[e3]; n_out[o+8] = enz[e3];

                        o += 9;
                        ti += 3;
                    }
                }
            }
        }
    }
    o as i32
    """
end

# Zero-copy payload splicing
@AX.rust_fn function _rust_build_payload!(
    v::Ptr{Float32}, n::Ptr{Float32}, vc::Int32, dest::Ptr{Float32}
)::Cvoid
    """
    let vc = vc as usize;
    unsafe {
        *dest = vc as f32;
        std::ptr::copy_nonoverlapping(v, dest.add(1), vc);
        std::ptr::copy_nonoverlapping(n, dest.add(1 + vc), vc);
    }
    """
end

#=
WGSL Compute Shader
=#
const _FIELD_WGSL = """
struct BlobData {
    x: f32,
    y: f32,
    z: f32,
    size: f32,
}

struct Params {
    ball_count: u32,
    grid_resolution: u32,
    subtract: f32,
    epsilon: f32,
}

@group(0) @binding(0) var<storage, read_write> field : array<f32>;
@group(0) @binding(1) var<storage, read>       axis  : array<f32>;
@group(0) @binding(2) var<storage, read>       blobs : array<BlobData>;
@group(0) @binding(3) var<uniform>            params : Params;

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
    let ix = global_id.x;
    let iy = global_id.y;
    let iz = global_id.z;
    let res = params.grid_resolution;
    
    if (ix >= res || iy >= res || iz >= res) { return; }
    
    let px = axis[ix];
    let py = axis[iy];
    let pz = axis[iz];
    var value = 0.0;
    
    for (var i = 0u; i < params.ball_count; i = i + 1u) {
        let blob = blobs[i];
        let dx = px - blob.x;
        let dy = py - blob.y;
        let dz = pz - blob.z;
        let contribution = blob.size / (params.epsilon + dx * dx + dy * dy + dz * dz) - params.subtract;
        if (contribution > 0.0) {
            value = value + contribution;
        }
    }
    
    field[ix + iy * res + iz * res * res] = value;
}
"""

#=
Julia Control Flow
=#
function init_blobs!()
    for i in 1:BALL_COUNT
        BLOB_X[i] = rand(Float32) * 0.6f0 + 0.2f0
        BLOB_Y[i] = rand(Float32) * 0.6f0 + 0.2f0
        BLOB_Z[i] = rand(Float32) * 0.6f0 + 0.2f0
        BLOB_VX[i] = (rand(Float32) - 0.5f0) * 0.005f0
        BLOB_VY[i] = (rand(Float32) - 0.5f0) * 0.005f0
        BLOB_VZ[i] = (rand(Float32) - 0.5f0) * 0.005f0
        BLOB_SIZE[i] = 0.45f0 + rand(Float32) * 0.1f0
    end
end

function init!()
    init_blobs!()
    AX.wgpu_init!()
    
    #=
    Before running on WGPU, first import the Julia-side EDGE_TABLE and TRI_TABLE into Rust.

    This completely eliminates the need to manually hardcode thousands of numbers in Rust strings.

    Because Julia's arrays are one-dimensional and continuous in memory, they can be converted to Int32 pointers and passed directly.
    =#
    edge_i32 = Int32.(EDGE_TABLE)
    tri_i32  = Int32.(TRI_TABLE)
    @AX.call_rust_fn _rust_init_tables!(pointer(edge_i32), pointer(tri_i32))

    AX.wgpu_create_buffer!(_BUF_FIELD,     GRID_SIZE * 4,       AX.BINDING_STORAGE_READ_WRITE)
    AX.wgpu_create_buffer!(_BUF_GRID_AXIS, GRID_RESOLUTION * 4, AX.BINDING_STORAGE_READ)
    AX.wgpu_create_buffer!(_BUF_BLOB_DATA, BALL_COUNT * 16,     AX.BINDING_STORAGE_READ)
    AX.wgpu_create_buffer!(_BUF_PARAMS,    16,                  AX.BINDING_UNIFORM)
    
    AX.wgpu_write_buffer!(_BUF_GRID_AXIS, GRID_AXIS)
    
    @AX.call_rust_fn _rust_pack_blobs!(pointer(BLOB_X), pointer(BLOB_Y), pointer(BLOB_Z), pointer(BLOB_SIZE), Int32(BALL_COUNT), pointer(_BLOB_DATA_BUF))
    AX.wgpu_write_buffer!(_BUF_BLOB_DATA, _BLOB_DATA_BUF)
    
    params_data = UInt32[BALL_COUNT, GRID_RESOLUTION]
    copyto!(_PARAMS_BUF, 1, reinterpret(UInt8, params_data), 1, 8)
    copyto!(_PARAMS_BUF, 9, reinterpret(UInt8, [SUBTRACT, FIELD_EPSILON]), 1, 8)
    AX.wgpu_write_buffer!(_BUF_PARAMS, _PARAMS_BUF)
    
    binding_flags = UInt32[AX.BINDING_STORAGE_READ_WRITE, AX.BINDING_STORAGE_READ, AX.BINDING_STORAGE_READ, AX.BINDING_UNIFORM]
    AX.wgpu_create_compute_pipeline!(_PIPELINE_ID, _FIELD_WGSL, "main", binding_flags)
    AX.wgpu_bind_buffers!(_PIPELINE_ID, [_BUF_FIELD, _BUF_GRID_AXIS, _BUF_BLOB_DATA, _BUF_PARAMS])
    
    @info "MetaballsAX ( Fully Optimized via Axis ) Initialized" backend="wgpu + Rust FFI"
end

function update_physics!(t::Float64)
    is_gravity_on = (t % 10) < 5 ? Int32(1) : Int32(0)
    @AX.call_rust_fn _rust_update_physics!(
        pointer(BLOB_X), pointer(BLOB_Y), pointer(BLOB_Z),
        pointer(BLOB_VX), pointer(BLOB_VY), pointer(BLOB_VZ),
        pointer(BLOB_SIZE), Int32(BALL_COUNT), is_gravity_on, SPEED_LIMIT
    )
end

function compute_field!(field::Vector{Float32})
    @AX.call_rust_fn _rust_pack_blobs!(pointer(BLOB_X), pointer(BLOB_Y), pointer(BLOB_Z), pointer(BLOB_SIZE), Int32(BALL_COUNT), pointer(_BLOB_DATA_BUF))
    AX.wgpu_write_buffer!(_BUF_BLOB_DATA, _BLOB_DATA_BUF)
    
    AX.wgpu_dispatch!(_PIPELINE_ID; wg_x=cld(GRID_RESOLUTION,8), wg_y=cld(GRID_RESOLUTION,8), wg_z=cld(GRID_RESOLUTION,4))
    AX.wgpu_read_buffer!(_BUF_FIELD, field)
    field
end

function build_mesh!(v::Vector{Float32}, n::Vector{Float32}, field::Vector{Float32})
    vc = @AX.call_rust_fn _rust_build_mesh!(
        pointer(v), pointer(n), pointer(field), pointer(GRID_AXIS), Int32(GRID_RESOLUTION),
        pointer(BLOB_X), pointer(BLOB_Y), pointer(BLOB_Z), pointer(BLOB_SIZE), Int32(BALL_COUNT),
        ISOLEVEL, SUBTRACT, FIELD_EPSILON
    )
    return Int(vc)
end

function build_payload!(v::Vector{Float32}, n::Vector{Float32}, vc::Int)
    @AX.call_rust_fn _rust_build_payload!(pointer(v), pointer(n), Int32(vc), pointer(PAYLOAD_BUFFER))
    return reinterpret(UInt8, @view PAYLOAD_BUFFER[1:(1 + vc * 2)])
end

end # module MetaballsAX