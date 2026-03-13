pub mod cpu_sa_summer;
pub mod cpu_sa_summer_fixed;
pub mod cpu_sa_winter;
pub mod cpu_sa_winter_fixed;
pub mod gpu_setup;
pub mod gpu_types;
pub mod gpu_types_summer;
pub mod gpu_types_summer_fixed;
pub mod gpu_types_winter;
pub mod gpu_types_winter_fixed;
pub mod output;
pub mod output_summer;
pub mod output_summer_fixed;
pub mod output_winter;
pub mod output_winter_fixed;
pub mod summer_main;
pub mod summer_fixed_main;
pub mod winter_main;
pub mod winter_fixed_main;
pub mod gpu_sa_loop;
pub mod island_pool;
pub mod winter_elite_main;
pub mod output_winter_elite;

#[cfg(test)]
mod test_histogram;

pub use solver_core::winter::*;
