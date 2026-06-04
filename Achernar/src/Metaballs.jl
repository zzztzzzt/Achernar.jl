module Metaballs

using Base.Threads
using CUDA
using StaticArrays

#=
Public API
=#
export FRAME_INTERVAL, ENABLE_THREADED_FIELD
export FIELD_BUFFER, VERTEX_BUFFER, NORMAL_BUFFER, PAYLOAD_BUFFER
export init_buffers!
export update_physics!, compute_field!, build_mesh!, build_payload!

#=
Constants
=#
const BALL_COUNT        = 12
const FRAME_INTERVAL    = 1 / 30
const SPEED_LIMIT       = 0.008f0

const GRID_RESOLUTION   = 108
const GRID_SIZE         = GRID_RESOLUTION * GRID_RESOLUTION * GRID_RESOLUTION
const CUBE_COUNT        = (GRID_RESOLUTION - 1) * (GRID_RESOLUTION - 1) * (GRID_RESOLUTION - 1)
const MAX_TRIANGLES     = CUBE_COUNT * 5
const ISOLEVEL          = 80.0f0
const SUBTRACT          = 8.0f0
const FIELD_EPSILON     = 1.0f-6
const THREAD_SLOT_COUNT = max(1, Threads.maxthreadid())

const ENABLE_THREADED_FIELD = nthreads() > 1
const ENABLE_THREADED_MESH  = nthreads() > 1

const GRID_AXIS = Float32[
    (i - 1) / Float32(GRID_RESOLUTION - 1)
    for i in 1:GRID_RESOLUTION
]

const EDGE_VERTEX_INDICES = (
    (1, 2), (2, 3), (3, 4), (4, 1),
    (5, 6), (6, 7), (7, 8), (8, 5),
    (1, 5), (2, 6), (3, 7), (4, 8),
)

#=
GPU buffers — Ref{Any}(nothing) avoids allocating GPU memory during precompilation.
Populated by init_gpu_buffers!() at runtime.
=#
const _CUDA_ENABLED  = Ref(false)
const d_GRID_AXIS    = Ref{Any}(nothing)
const d_FIELD_BUFFER = Ref{Any}(nothing)
const d_BLOB_X       = Ref{Any}(nothing)
const d_BLOB_Y       = Ref{Any}(nothing)
const d_BLOB_Z       = Ref{Any}(nothing)
const d_BLOB_SIZE    = Ref{Any}(nothing)

include("utils/MarchingCubesTables.jl")

mutable struct Blob
    x::Float32
    y::Float32
    z::Float32
    vx::Float32
    vy::Float32
    vz::Float32
    size::Float32
end

function init_blobs()
    blobs = Blob[]
    for _ in 1:BALL_COUNT
        push!(blobs, Blob(
            rand(Float32) * 0.6f0 + 0.2f0,
            rand(Float32) * 0.6f0 + 0.2f0,
            rand(Float32) * 0.6f0 + 0.2f0,
            (rand(Float32) - 0.5f0) * 0.005f0,
            (rand(Float32) - 0.5f0) * 0.005f0,
            (rand(Float32) - 0.5f0) * 0.005f0,
            0.45f0 + rand(Float32) * 0.1f0,
        ))
    end
    blobs
end

const BLOBS = init_blobs()

# These CPU-side SoA buffers are reused every frame.
# We use Ref{Any}(nothing) to prevent 900MB of uninitialized memory 
# from being serialized into the precompiled .dll.
const BLOB_X_BUFFER    = Ref{Vector{Float32}}(Float32[])
const BLOB_Y_BUFFER    = Ref{Vector{Float32}}(Float32[])
const BLOB_Z_BUFFER    = Ref{Vector{Float32}}(Float32[])
const BLOB_SIZE_BUFFER = Ref{Vector{Float32}}(Float32[])
const FIELD_BUFFER     = Ref{Vector{Float32}}(Float32[])
const VERTEX_BUFFER    = Ref{Vector{Float32}}(Float32[])
const NORMAL_BUFFER    = Ref{Vector{Float32}}(Float32[])
const PAYLOAD_BUFFER   = Ref{Vector{Float32}}(Float32[])

const THREAD_VERTEX_BUFFERS = Ref{Vector{Vector{Float32}}}(Vector{Float32}[])
const THREAD_NORMAL_BUFFERS = Ref{Vector{Vector{Float32}}}(Vector{Float32}[])
const THREAD_MESH_COUNTS    = Ref{Vector{Int}}(Int[])
const THREAD_EDGE_X         = Ref{Vector{Vector{Float32}}}(Vector{Float32}[])
const THREAD_EDGE_Y         = Ref{Vector{Vector{Float32}}}(Vector{Float32}[])
const THREAD_EDGE_Z         = Ref{Vector{Vector{Float32}}}(Vector{Float32}[])
const THREAD_EDGE_NX        = Ref{Vector{Vector{Float32}}}(Vector{Float32}[])
const THREAD_EDGE_NY        = Ref{Vector{Vector{Float32}}}(Vector{Float32}[])
const THREAD_EDGE_NZ        = Ref{Vector{Vector{Float32}}}(Vector{Float32}[])

function init_buffers!()
    # Initialize CPU Buffers
    BLOB_X_BUFFER[]    = zeros(Float32, BALL_COUNT)
    BLOB_Y_BUFFER[]    = zeros(Float32, BALL_COUNT)
    BLOB_Z_BUFFER[]    = zeros(Float32, BALL_COUNT)
    BLOB_SIZE_BUFFER[] = zeros(Float32, BALL_COUNT)
    FIELD_BUFFER[]     = Vector{Float32}(undef, GRID_SIZE)
    VERTEX_BUFFER[]    = Vector{Float32}(undef, MAX_TRIANGLES * 9)
    NORMAL_BUFFER[]    = Vector{Float32}(undef, MAX_TRIANGLES * 9)
    PAYLOAD_BUFFER[]   = Vector{Float32}(undef, 1 + MAX_TRIANGLES * 18)

    thread_slot_count = max(1, Threads.maxthreadid())
    thread_mesh_capacity = max(9, cld(MAX_TRIANGLES * 9, thread_slot_count) * 2)

    THREAD_VERTEX_BUFFERS[] = [Vector{Float32}(undef, thread_mesh_capacity) for _ in 1:thread_slot_count]
    THREAD_NORMAL_BUFFERS[] = [Vector{Float32}(undef, thread_mesh_capacity) for _ in 1:thread_slot_count]
    THREAD_MESH_COUNTS[]    = zeros(Int, thread_slot_count)
    THREAD_EDGE_X[]  = [Vector{Float32}(undef, 12) for _ in 1:thread_slot_count]
    THREAD_EDGE_Y[]  = [Vector{Float32}(undef, 12) for _ in 1:thread_slot_count]
    THREAD_EDGE_Z[]  = [Vector{Float32}(undef, 12) for _ in 1:thread_slot_count]
    THREAD_EDGE_NX[] = [Vector{Float32}(undef, 12) for _ in 1:thread_slot_count]
    THREAD_EDGE_NY[] = [Vector{Float32}(undef, 12) for _ in 1:thread_slot_count]
    THREAD_EDGE_NZ[] = [Vector{Float32}(undef, 12) for _ in 1:thread_slot_count]

    # Initialize GPU Buffers
    _CUDA_ENABLED[] = try
        CUDA.functional()
    catch
        false
    end

    if _CUDA_ENABLED[]
        d_GRID_AXIS[]    = CuArray(GRID_AXIS)
        d_FIELD_BUFFER[] = CUDA.zeros(Float32, GRID_SIZE)
        d_BLOB_X[]       = CUDA.zeros(Float32, BALL_COUNT)
        d_BLOB_Y[]       = CUDA.zeros(Float32, BALL_COUNT)
        d_BLOB_Z[]       = CUDA.zeros(Float32, BALL_COUNT)
        d_BLOB_SIZE[]    = CUDA.zeros(Float32, BALL_COUNT)
    end
end

@inline grid_index(x::Int, y::Int, z::Int) = x + (y - 1) * GRID_RESOLUTION + (z - 1) * GRID_RESOLUTION * GRID_RESOLUTION

# CUDA kernel for scalar-field evaluation only. Mesh extraction stays on the CPU for stability
function field_kernel!(field, axis, bx, by, bz, bsize)
    ix = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    iy = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    iz = (blockIdx().z - 1) * blockDim().z + threadIdx().z

    if ix <= GRID_RESOLUTION && iy <= GRID_RESOLUTION && iz <= GRID_RESOLUTION
        px = axis[ix]
        py = axis[iy]
        pz = axis[iz]
        value = 0.0f0

        for i in 1:BALL_COUNT
            dx = px - bx[i]
            dy = py - by[i]
            dz = pz - bz[i]
            contribution = bsize[i] / (FIELD_EPSILON + dx * dx + dy * dy + dz * dz) - SUBTRACT
            if contribution > 0.0f0
                value += contribution
            end
        end

        field[ix + (iy - 1) * GRID_RESOLUTION + (iz - 1) * GRID_RESOLUTION * GRID_RESOLUTION] = value
    end
    return nothing
end

function update_physics!(current_time::Float64)
    # Alternate between attraction and repulsion so the blobs keep reconfiguring over time
    is_gravity_on = (current_time % 10) < 5

    for i in 1:BALL_COUNT
        bi = BLOBS[i]
        bi.x += bi.vx
        bi.y += bi.vy
        bi.z += bi.vz

        for j in 1:BALL_COUNT
            i == j && continue

            bj = BLOBS[j]
            dx = bj.x - bi.x
            dy = bj.y - bi.y
            dz = bj.z - bi.z
            dist_sq = dx^2 + dy^2 + dz^2 + 0.01f0

            if is_gravity_on
                force = 0.000008f0 / dist_sq
                bi.vx += dx * force
                bi.vy += dy * force
                bi.vz += dz * force
            else
                push_force = 0.00002f0 / dist_sq
                bi.vx -= dx * push_force
                bi.vy -= dy * push_force
                bi.vz -= dz * push_force
            end
        end

        margin = 0.15f0
        if bi.x < margin
            bi.vx += 0.001f0
        elseif bi.x > 1.0f0 - margin
            bi.vx -= 0.001f0
        end

        if bi.y < margin
            bi.vy += 0.001f0
        elseif bi.y > 1.0f0 - margin
            bi.vy -= 0.001f0
        end

        if bi.z < margin
            bi.vz += 0.001f0
        elseif bi.z > 1.0f0 - margin
            bi.vz -= 0.001f0
        end

        bi.vx = clamp(bi.vx * 0.98f0, -SPEED_LIMIT, SPEED_LIMIT)
        bi.vy = clamp(bi.vy * 0.98f0, -SPEED_LIMIT, SPEED_LIMIT)
        bi.vz = clamp(bi.vz * 0.98f0, -SPEED_LIMIT, SPEED_LIMIT)
    end
end

function update_blob_buffers!()
    # Copy the mutable blob structs into flat arrays before the GPU upload
    @inbounds for i in 1:BALL_COUNT
        blob = BLOBS[i]
        BLOB_X_BUFFER[][i]    = blob.x
        BLOB_Y_BUFFER[][i]    = blob.y
        BLOB_Z_BUFFER[][i]    = blob.z
        BLOB_SIZE_BUFFER[][i] = blob.size
    end
end

function compute_field_serial!(field::Vector{Float32})
    @inbounds for z in 1:GRID_RESOLUTION
        pz = GRID_AXIS[z]
        for y in 1:GRID_RESOLUTION
            py = GRID_AXIS[y]
            for x in 1:GRID_RESOLUTION
                px = GRID_AXIS[x]
                value = 0.0f0

                for blob in BLOBS
                    dx = px - blob.x
                    dy = py - blob.y
                    dz = pz - blob.z
                    contribution = blob.size / (FIELD_EPSILON + dx * dx + dy * dy + dz * dz) - SUBTRACT
                    if contribution > 0.0f0
                        value += contribution
                    end
                end

                field[grid_index(x, y, z)] = value
            end
        end
    end
end

function compute_field_threaded!(field::Vector{Float32})
    # Each z-slice is independent, so field sampling parallelizes cleanly on the CPU
    @threads for z in 1:GRID_RESOLUTION
        pz = GRID_AXIS[z]
        @inbounds for y in 1:GRID_RESOLUTION
            py = GRID_AXIS[y]
            for x in 1:GRID_RESOLUTION
                px = GRID_AXIS[x]
                value = 0.0f0

                for blob in BLOBS
                    dx = px - blob.x
                    dy = py - blob.y
                    dz = pz - blob.z
                    contribution = blob.size / (FIELD_EPSILON + dx * dx + dy * dy + dz * dz) - SUBTRACT
                    if contribution > 0.0f0
                        value += contribution
                    end
                end

                field[grid_index(x, y, z)] = value
            end
        end
    end
end

function compute_field_gpu!(field::Vector{Float32})
    update_blob_buffers!()
    copyto!(d_BLOB_X[], BLOB_X_BUFFER[])
    copyto!(d_BLOB_Y[], BLOB_Y_BUFFER[])
    copyto!(d_BLOB_Z[], BLOB_Z_BUFFER[])
    copyto!(d_BLOB_SIZE[], BLOB_SIZE_BUFFER[])

    threads = (8, 8, 8)
    blocks = (
        cld(GRID_RESOLUTION, threads[1]),
        cld(GRID_RESOLUTION, threads[2]),
        cld(GRID_RESOLUTION, threads[3]),
    )
    @cuda threads=threads blocks=blocks field_kernel!(d_FIELD_BUFFER[], d_GRID_AXIS[], d_BLOB_X[], d_BLOB_Y[], d_BLOB_Z[], d_BLOB_SIZE[])
    CUDA.synchronize()
    copyto!(field, d_FIELD_BUFFER[])
end

function compute_field!(field::Vector{Float32})
    if _CUDA_ENABLED[]
        compute_field_gpu!(field)
    elseif ENABLE_THREADED_FIELD
        compute_field_threaded!(field)
    else
        compute_field_serial!(field)
    end
end

@inline function sample_gradient(x::Float32, y::Float32, z::Float32)
    gx = 0.0f0
    gy = 0.0f0
    gz = 0.0f0

    @inbounds for blob in BLOBS
        dx = x - blob.x
        dy = y - blob.y
        dz = z - blob.z
        dist_sq = FIELD_EPSILON + dx * dx + dy * dy + dz * dz
        contribution = blob.size / dist_sq - SUBTRACT

        if contribution > 0.0f0
            scale = -2.0f0 * blob.size / (dist_sq * dist_sq)
            gx += dx * scale
            gy += dy * scale
            gz += dz * scale
        end
    end

    len = sqrt(gx * gx + gy * gy + gz * gz)
    if len > 1.0f-6
        inv_len = 1.0f0 / len
        return (gx * inv_len, gy * inv_len, gz * inv_len)
    end

    return (0.0f0, 1.0f0, 0.0f0)
end

@inline function interpolate_vertex(
    ax::Float32, ay::Float32, az::Float32, av::Float32,
    bx::Float32, by::Float32, bz::Float32, bv::Float32,
)
    delta = bv - av
    t = abs(delta) < 1.0f-6 ? 0.5f0 : clamp((ISOLEVEL - av) / delta, 0.0f0, 1.0f0)
    return (
        ax + (bx - ax) * t,
        ay + (by - ay) * t,
        az + (bz - az) * t,
    )
end

@inline function append_triangle!(
    vertices::Vector{Float32},
    normals::Vector{Float32},
    offset::Int,
    ax::Float32, ay::Float32, az::Float32,
    bx::Float32, by::Float32, bz::Float32,
    cx::Float32, cy::Float32, cz::Float32,
    anx::Float32, any::Float32, anz::Float32,
    bnx::Float32, bny::Float32, bnz::Float32,
    cnx::Float32, cny::Float32, cnz::Float32,
) 
    vertices[offset]     = ax * 2.0f0 - 1.0f0
    vertices[offset + 1] = ay * 2.0f0 - 1.0f0
    vertices[offset + 2] = az * 2.0f0 - 1.0f0
    vertices[offset + 3] = bx * 2.0f0 - 1.0f0
    vertices[offset + 4] = by * 2.0f0 - 1.0f0
    vertices[offset + 5] = bz * 2.0f0 - 1.0f0
    vertices[offset + 6] = cx * 2.0f0 - 1.0f0
    vertices[offset + 7] = cy * 2.0f0 - 1.0f0
    vertices[offset + 8] = cz * 2.0f0 - 1.0f0

    normals[offset]     = anx
    normals[offset + 1] = any
    normals[offset + 2] = anz
    normals[offset + 3] = bnx
    normals[offset + 4] = bny
    normals[offset + 5] = bnz
    normals[offset + 6] = cnx
    normals[offset + 7] = cny
    normals[offset + 8] = cnz

    return offset + 9
end

function build_mesh!(vertices::Vector{Float32}, normals::Vector{Float32}, field::Vector{Float32})
    if ENABLE_THREADED_MESH
        return build_mesh_threaded!(vertices, normals, field)
    end
    return build_mesh_serial!(vertices, normals, field)
end

function build_mesh_serial!(vertices::Vector{Float32}, normals::Vector{Float32}, field::Vector{Float32})
    offset = 1
    edge_x = Vector{Float32}(undef, 12)
    edge_y = Vector{Float32}(undef, 12)
    edge_z = Vector{Float32}(undef, 12)
    edge_nx = Vector{Float32}(undef, 12)
    edge_ny = Vector{Float32}(undef, 12)
    edge_nz = Vector{Float32}(undef, 12)

    @inbounds for z in 1:(GRID_RESOLUTION - 1)
        z0 = GRID_AXIS[z]
        z1 = GRID_AXIS[z + 1]

        for y in 1:(GRID_RESOLUTION - 1)
            y0 = GRID_AXIS[y]
            y1 = GRID_AXIS[y + 1]

            for x in 1:(GRID_RESOLUTION - 1)
                x0 = GRID_AXIS[x]
                x1 = GRID_AXIS[x + 1]

                q    = grid_index(x,     y,     z)
                q1   = grid_index(x + 1, y,     z)
                qy   = grid_index(x,     y + 1, z)
                q1y  = grid_index(x + 1, y + 1, z)
                qz   = grid_index(x,     y,     z + 1)
                q1z  = grid_index(x + 1, y,     z + 1)
                qyz  = grid_index(x,     y + 1, z + 1)
                q1yz = grid_index(x + 1, y + 1, z + 1)

                cube_values = SVector{8, Float32}(
                    field[q], field[q1], field[q1y], field[qy],
                    field[qz], field[q1z], field[q1yz], field[qyz]
                )

                cube_index = 0
                for i in 1:8
                    if cube_values[i] < ISOLEVEL
                        cube_index |= 1 << (i - 1)
                    end
                end

                edge_mask = EDGE_TABLE[cube_index + 1]
                edge_mask == 0 && continue

                cube_x = SVector{8, Float32}(x0, x1, x1, x0, x0, x1, x1, x0)
                cube_y = SVector{8, Float32}(y0, y0, y1, y1, y0, y0, y1, y1)
                cube_z = SVector{8, Float32}(z0, z0, z0, z0, z1, z1, z1, z1)

                for edge in 1:12
                    if (edge_mask & (1 << (edge - 1))) == 0
                        continue
                    end

                    ia, ib = EDGE_VERTEX_INDICES[edge]
                    vx, vy, vz = interpolate_vertex(
                        cube_x[ia], cube_y[ia], cube_z[ia], cube_values[ia],
                        cube_x[ib], cube_y[ib], cube_z[ib], cube_values[ib],
                    )
                    edge_x[edge] = vx
                    edge_y[edge] = vy
                    edge_z[edge] = vz
                    
                    nx, ny, nz = sample_gradient(vx, vy, vz)
                    edge_nx[edge] = nx
                    edge_ny[edge] = ny
                    edge_nz[edge] = nz
                end

                tri_idx = cube_index * 16 + 1

                while TRI_TABLE[tri_idx] != -1
                    e1 = TRI_TABLE[tri_idx] + 1
                    e2 = TRI_TABLE[tri_idx + 1] + 1
                    e3 = TRI_TABLE[tri_idx + 2] + 1

                    offset = append_triangle!(
                        vertices, normals, offset,
                        edge_x[e1], edge_y[e1], edge_z[e1],
                        edge_x[e2], edge_y[e2], edge_z[e2],
                        edge_x[e3], edge_y[e3], edge_z[e3],
                        edge_nx[e1], edge_ny[e1], edge_nz[e1],
                        edge_nx[e2], edge_ny[e2], edge_nz[e2],
                        edge_nx[e3], edge_ny[e3], edge_nz[e3],
                    )

                    tri_idx += 3
                end
            end
        end
    end

    return offset - 1
end

function build_mesh_threaded!(vertices::Vector{Float32}, normals::Vector{Float32}, field::Vector{Float32})
    fill!(THREAD_MESH_COUNTS[], 0)

    @threads for z in 1:(GRID_RESOLUTION - 1)
        tid = threadid()
        local_vertices = THREAD_VERTEX_BUFFERS[][tid]
        local_normals  = THREAD_NORMAL_BUFFERS[][tid]
        edge_x = THREAD_EDGE_X[][tid]
        edge_y = THREAD_EDGE_Y[][tid]
        edge_z = THREAD_EDGE_Z[][tid]
        edge_nx = THREAD_EDGE_NX[][tid]
        edge_ny = THREAD_EDGE_NY[][tid]
        edge_nz = THREAD_EDGE_NZ[][tid]
        offset = THREAD_MESH_COUNTS[][tid] + 1

        @inbounds begin
            z0 = GRID_AXIS[z]
            z1 = GRID_AXIS[z + 1]

            for y in 1:(GRID_RESOLUTION - 1)
                y0 = GRID_AXIS[y]
                y1 = GRID_AXIS[y + 1]

                for x in 1:(GRID_RESOLUTION - 1)
                    x0 = GRID_AXIS[x]
                    x1 = GRID_AXIS[x + 1]

                    q    = grid_index(x,     y,     z)
                    q1   = grid_index(x + 1, y,     z)
                    qy   = grid_index(x,     y + 1, z)
                    q1y  = grid_index(x + 1, y + 1, z)
                    qz   = grid_index(x,     y,     z + 1)
                    q1z  = grid_index(x + 1, y,     z + 1)
                    qyz  = grid_index(x,     y + 1, z + 1)
                    q1yz = grid_index(x + 1, y + 1, z + 1)

                    # Use SVector ( this is 100% configured on the register, zero heap allocation )
                    cube_values = SVector{8, Float32}(
                        field[q], field[q1], field[q1y], field[qy],
                        field[qz], field[q1z], field[q1yz], field[qyz]
                    )

                    cube_index = 0
                    for i in 1:8
                        if cube_values[i] < ISOLEVEL
                            cube_index |= 1 << (i - 1)
                        end
                    end

                    edge_mask = EDGE_TABLE[cube_index + 1]
                    edge_mask == 0 && continue

                    cube_x = SVector{8, Float32}(x0, x1, x1, x0, x0, x1, x1, x0)
                    cube_y = SVector{8, Float32}(y0, y0, y1, y1, y0, y0, y1, y1)
                    cube_z = SVector{8, Float32}(z0, z0, z0, z0, z1, z1, z1, z1)

                    # Boundary interpolation and normal vector estimation ( calculated only once per edge )
                    for edge in 1:12
                        if (edge_mask & (1 << (edge - 1))) == 0
                            continue
                        end

                        ia, ib = EDGE_VERTEX_INDICES[edge]
                        vx, vy, vz = interpolate_vertex(
                            cube_x[ia], cube_y[ia], cube_z[ia], cube_values[ia],
                            cube_x[ib], cube_y[ib], cube_z[ib], cube_values[ib],
                        )
                        edge_x[edge] = vx
                        edge_y[edge] = vy
                        edge_z[edge] = vz
                        
                        # Once the vertex is calculated, immediately calculate its normal vector and cache it
                        nx, ny, nz = sample_gradient(vx, vy, vz)
                        edge_nx[edge] = nx
                        edge_ny[edge] = ny
                        edge_nz[edge] = nz
                    end

                    tri_idx = cube_index * 16 + 1

                    # Triangle generation ( purely table lookup, extremely fast )
                    while TRI_TABLE[tri_idx] != -1
                        offset + 8 <= length(local_vertices) || error("Thread-local mesh buffer overflow.")
                        
                        e1 = TRI_TABLE[tri_idx] + 1
                        e2 = TRI_TABLE[tri_idx + 1] + 1
                        e3 = TRI_TABLE[tri_idx + 2] + 1

                        offset = append_triangle!(
                            local_vertices, local_normals, offset,
                            edge_x[e1], edge_y[e1], edge_z[e1],
                            edge_x[e2], edge_y[e2], edge_z[e2],
                            edge_x[e3], edge_y[e3], edge_z[e3],
                            edge_nx[e1], edge_ny[e1], edge_nz[e1], # Directly substitute from table
                            edge_nx[e2], edge_ny[e2], edge_nz[e2],
                            edge_nx[e3], edge_ny[e3], edge_nz[e3],
                        )
                        tri_idx += 3
                    end
                end
            end
        end

        THREAD_MESH_COUNTS[][tid] = offset - 1
    end

    offset = 1
    for tid in 1:THREAD_SLOT_COUNT
        count = THREAD_MESH_COUNTS[][tid]
        count == 0 && continue
        offset + count - 1 <= length(vertices) || error("Global mesh buffer overflow.")
        copyto!(vertices, offset, THREAD_VERTEX_BUFFERS[][tid], 1, count)
        copyto!(normals,  offset, THREAD_NORMAL_BUFFERS[][tid], 1, count)
        offset += count
    end

    return offset - 1
end

function build_payload!(
    payload::Vector{Float32},
    vertices::Vector{Float32},
    normals::Vector{Float32},
    vertex_float_count::Int,
)
    payload[1] = Float32(vertex_float_count)
    copyto!(payload, 2, vertices, 1, vertex_float_count)
    copyto!(payload, 2 + vertex_float_count, normals, 1, vertex_float_count)
    return reinterpret(UInt8, @view payload[1:(1 + vertex_float_count * 2)])
end

end # module Metaballs
