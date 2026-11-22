use bevy::asset::RenderAssetUsages;
use bevy::mesh::{Indices, Mesh, MeshVertexBufferLayoutRef};
use bevy::pbr::{Material, MaterialPipeline, MaterialPipelineKey};
use bevy::prelude::*;
use bevy::render::render_resource::{
    AsBindGroup, PrimitiveTopology, RenderPipelineDescriptor, ShaderType,
    SpecializedMeshPipelineError,
};
use bevy::render::storage::ShaderBuffer;
use bevy::shader::ShaderRef;
use liquid_wave::grid::{Bathymetry, Grid, State};

pub const HEIGHTFIELD_SHADER: &str = "embedded://liquid_ocean/heightfield_surface.wesl";
pub const OCEAN_SHADER: &str = "embedded://liquid_ocean/ocean_surface.wesl";

#[derive(Clone, Copy, Debug, ShaderType)]
pub struct WaterLook {
    pub absorption: Vec4,
    pub scatter: Vec4,
    pub sky_zenith: Vec4,
    pub sky_horizon: Vec4,
    pub sun: Vec4,
    pub foam: Vec4,
}

impl Default for WaterLook {
    fn default() -> Self {
        Self {
            absorption: Vec4::new(0.45, 0.075, 0.045, 0.0),
            scatter: Vec4::new(0.012, 0.075, 0.095, 0.0),
            sky_zenith: Vec4::new(0.18, 0.36, 0.72, 0.0),
            sky_horizon: Vec4::new(0.62, 0.74, 0.86, 0.0),
            sun: Vec4::new(0.45, 0.55, 0.35, 60.0),
            foam: Vec4::new(0.92, 0.95, 0.97, 1.0),
        }
    }
}

#[derive(Clone, Copy, Debug, ShaderType)]
pub struct HeightfieldSettings {
    pub size: UVec4,
    pub cell: Vec4,
    pub look: WaterLook,
}

#[derive(Asset, TypePath, AsBindGroup, Clone, Debug)]
pub struct HeightfieldMaterial {
    #[uniform(0)]
    pub settings: HeightfieldSettings,
    #[storage(1, read_only)]
    pub state: Handle<ShaderBuffer>,
    #[storage(2, read_only)]
    pub bed: Handle<ShaderBuffer>,
}

impl Material for HeightfieldMaterial {
    fn vertex_shader() -> ShaderRef {
        HEIGHTFIELD_SHADER.into()
    }

    fn fragment_shader() -> ShaderRef {
        HEIGHTFIELD_SHADER.into()
    }

    fn reads_view_transmission_texture(&self) -> bool {
        true
    }

    fn enable_prepass() -> bool {
        false
    }

    fn enable_shadows() -> bool {
        false
    }

    fn specialize(
        _pipeline: &MaterialPipeline,
        descriptor: &mut RenderPipelineDescriptor,
        _layout: &MeshVertexBufferLayoutRef,
        _key: MaterialPipelineKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        descriptor.primitive.cull_mode = None;
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, ShaderType)]
pub struct OceanSettings {
    pub grid: UVec4,
    pub patch_size: Vec4,
    pub params: Vec4,
    pub look: WaterLook,
}

#[derive(Asset, TypePath, AsBindGroup, Clone, Debug)]
pub struct OceanMaterial {
    #[uniform(0)]
    pub settings: OceanSettings,
    #[storage(1, read_only)]
    pub surface: Handle<ShaderBuffer>,
}

impl Material for OceanMaterial {
    fn vertex_shader() -> ShaderRef {
        OCEAN_SHADER.into()
    }

    fn fragment_shader() -> ShaderRef {
        OCEAN_SHADER.into()
    }

    fn reads_view_transmission_texture(&self) -> bool {
        true
    }

    fn enable_prepass() -> bool {
        false
    }

    fn enable_shadows() -> bool {
        false
    }

    fn specialize(
        _pipeline: &MaterialPipeline,
        descriptor: &mut RenderPipelineDescriptor,
        _layout: &MeshVertexBufferLayoutRef,
        _key: MaterialPipelineKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        descriptor.primitive.cull_mode = None;
        Ok(())
    }
}

fn grid_indices(columns: u32, rows: u32) -> Vec<u32> {
    let mut indices = Vec::with_capacity(((columns - 1) * (rows - 1) * 6) as usize);

    for k in 0..rows - 1 {
        for i in 0..columns - 1 {
            let a = k * columns + i;
            let b = a + 1;
            let c = a + columns;
            let d = c + 1;
            indices.extend([a, c, b, b, c, d]);
        }
    }

    indices
}

fn mesh_from(positions: Vec<[f32; 3]>, uvs: Vec<[f32; 2]>, indices: Vec<u32>) -> Mesh {
    let normals = vec![[0.0, 1.0, 0.0]; positions.len()];

    Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::RENDER_WORLD,
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
    .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, uvs)
    .with_inserted_indices(Indices::U32(indices))
}

pub fn heightfield_mesh(grid: &Grid, state: &State, bathymetry: &Bathymetry) -> Mesh {
    let mut positions = Vec::with_capacity(grid.cells());
    let mut uvs = Vec::with_capacity(grid.cells());

    for k in 0..grid.nz {
        for i in 0..grid.nx {
            let [x, z] = grid.center(i, k);
            let c = grid.cell(i, k);
            let y = bathymetry.cell(c).level(state.h[c]);
            positions.push([x as f32, y as f32, z as f32]);
            uvs.push([i as f32, k as f32]);
        }
    }

    mesh_from(positions, uvs, grid_indices(grid.nx as u32, grid.nz as u32))
}

pub fn ocean_mesh(radius: f32, resolution: u32) -> Mesh {
    let side = resolution + 1;
    let mut positions = Vec::with_capacity((side * side) as usize);

    for k in 0..side {
        for i in 0..side {
            let warp = |t: u32| {
                let s = 2.0 * t as f32 / resolution as f32 - 1.0;
                radius * s * s.abs()
            };
            positions.push([warp(i), 0.0, warp(k)]);
        }
    }

    let uvs = vec![[0.0, 0.0]; positions.len()];

    mesh_from(positions, uvs, grid_indices(side, side))
}
