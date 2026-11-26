pub fn portable(supported: wgpu::Limits) -> wgpu::Limits {
    supported.or_worse_values_from(&wgpu::Limits {
        max_storage_buffers_per_shader_stage: 29,
        max_buffers_and_acceleration_structures_per_shader_stage: 29,
        max_storage_textures_per_shader_stage: 32,
        max_storage_buffer_binding_size: 1 << 27,
        max_uniform_buffer_binding_size: 1 << 16,
        max_buffer_size: i32::MAX as u64,
        max_compute_workgroup_storage_size: 1 << 15,
        max_compute_invocations_per_workgroup: 1024,
        max_compute_workgroup_size_x: 1024,
        max_compute_workgroup_size_y: 1024,
        max_compute_workgroup_size_z: 1024,
        max_texture_dimension_1d: 1 << 14,
        max_texture_dimension_2d: 1 << 14,
        max_texture_dimension_3d: 1 << 12,
        max_texture_array_layers: 2048,
        ..wgpu::Limits::unlimited()
    })
}
