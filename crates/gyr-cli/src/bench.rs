//! `gyr bench`. Filled in by milestone 6.

use clap::Subcommand;
use gyrfalcon::Error;

#[derive(Subcommand)]
pub enum BenchCommand {
    /// Placeholder until the bench harness lands.
    Retrieve,
}

pub fn run(_command: BenchCommand) -> Result<(), Error> {
    Err(Error::Io(std::io::Error::other("bench is not implemented yet")))
}
