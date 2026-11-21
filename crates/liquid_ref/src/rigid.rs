use glam::{DMat3, DQuat, DVec3};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Shape {
    Sphere { radius: f64 },
    Cuboid { half: DVec3 },
}

impl Shape {
    pub fn sdf_local(&self, p: DVec3) -> f64 {
        match *self {
            Shape::Sphere { radius } => p.length() - radius,
            Shape::Cuboid { half } => {
                let q = p.abs() - half;
                q.max(DVec3::ZERO).length() + q.max_element().min(0.0)
            }
        }
    }

    pub fn volume(&self) -> f64 {
        match *self {
            Shape::Sphere { radius } => 4.0 / 3.0 * std::f64::consts::PI * radius.powi(3),
            Shape::Cuboid { half } => 8.0 * half.x * half.y * half.z,
        }
    }

    pub fn unit_inertia(&self) -> DVec3 {
        match *self {
            Shape::Sphere { radius } => DVec3::splat(0.4 * radius * radius),
            Shape::Cuboid { half } => {
                let s = half * 2.0;
                DVec3::new(
                    s.y * s.y + s.z * s.z,
                    s.x * s.x + s.z * s.z,
                    s.x * s.x + s.y * s.y,
                ) / 12.0
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Motion {
    Static,
    Kinematic,
    Dynamic,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RigidBody {
    pub shape: Shape,

    pub motion: Motion,

    pub position: DVec3,

    pub rotation: DQuat,

    pub velocity: DVec3,

    pub angular_velocity: DVec3,

    pub density: f64,

    pub lock_rotation: bool,
}

impl RigidBody {
    pub fn new(shape: Shape, position: DVec3, motion: Motion, density: f64) -> Self {
        Self {
            shape,
            motion,
            position,
            rotation: DQuat::IDENTITY,
            velocity: DVec3::ZERO,
            angular_velocity: DVec3::ZERO,
            density,
            lock_rotation: false,
        }
    }

    pub fn sdf(&self, x: DVec3) -> f64 {
        self.shape
            .sdf_local(self.rotation.inverse() * (x - self.position))
    }

    pub fn velocity_at(&self, x: DVec3) -> DVec3 {
        self.velocity + self.angular_velocity.cross(x - self.position)
    }

    pub fn mass(&self) -> f64 {
        self.density * self.shape.volume()
    }

    pub fn inverse_inertia_world(&self) -> DMat3 {
        if self.lock_rotation {
            return DMat3::ZERO;
        }

        let r = DMat3::from_quat(self.rotation);
        let local = DMat3::from_diagonal(1.0 / (self.shape.unit_inertia() * self.mass()));

        r * local * r.transpose()
    }

    pub fn is_dynamic(&self) -> bool {
        self.motion == Motion::Dynamic
    }

    pub fn integrate(&mut self, dt: f64) {
        if self.motion == Motion::Static {
            return;
        }

        self.position += self.velocity * dt;

        let w = self.angular_velocity;

        if !self.lock_rotation && w.length_squared() > 0.0 {
            let dq = DQuat::from_scaled_axis(w * dt);
            self.rotation = (dq * self.rotation).normalize();
        }
    }
}

pub fn apply_inverse_mass(body: &RigidBody, impulse: [f64; 6]) -> [f64; 6] {
    let inv_m = 1.0 / body.mass();
    let lin = DVec3::new(impulse[0], impulse[1], impulse[2]) * inv_m;
    let ang = body.inverse_inertia_world() * DVec3::new(impulse[3], impulse[4], impulse[5]);

    [lin.x, lin.y, lin.z, ang.x, ang.y, ang.z]
}
