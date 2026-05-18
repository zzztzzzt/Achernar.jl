#=
Julia Script to Generate Rust Bridge Code

Run this from your terminal/command line to generate Rust FFI code :
  julia --project=. scripts/generate_bridge.jl
=#

using Pkg
# Activate the parent workspace directory containing Achernar and Axis
Pkg.activate(abspath(joinpath(@__DIR__, "..")))

@info "Loading Achernar module..."
using Achernar

import Axis as AX
axis_generated_dir = abspath(joinpath(@__DIR__, "..", "axis_rs", "src", "generated"))

@info "Triggering Axis Rust code generator..." axis_generated_dir
AX.generate_bridge(axis_generated_dir)

@info "Generation complete! Rust files successfully written to: $axis_generated_dir"
