use commands::{search::search_db, serve::serve_search, updatedb::update_db};

use mmindex::{database::Database, embedding::CLIPModelInfo};

use std::{
    path::PathBuf,
    sync::{Arc, Mutex},
};

use serde_derive::Deserialize;

use clap::{CommandFactory, Parser, Subcommand};

pub mod commands {
    pub mod search;
    pub mod serve;
    pub mod updatedb;
}

pub mod mmindex {
    pub mod database;
    pub mod embedding;
    pub mod math;
    pub mod query;
}

#[derive(Deserialize, Debug)]
pub struct Config {
    model_info: CLIPModelInfo,
    db_path: PathBuf,
    gpu_id: u32,
}

#[derive(Parser)]
struct Args {
    #[arg(
        short,
        long,
        default_value = "./model_infos/ViT-L-16-SigLIP_webli.toml"
    )]
    model_config: PathBuf,
    #[arg(short, long, default_value = "./db")]
    db_path: PathBuf,
    #[arg(short, long, default_value = "0")]
    gpu_id: u32,
    #[command(subcommand)]
    command: Option<Commands>,
}
#[derive(Subcommand)]
enum Commands {
    UpdateDb {
        #[arg(short, long)]
        media_dirs: Vec<PathBuf>,
        #[arg(short, long, default_value = "8")]
        batch_size: usize,
        #[arg(short, long, default_value = "false")]
        tensorrt: bool,
        #[arg(short, long, default_value = "2")]
        readers: usize,
    },
    Serve,
    Search {
        query: String,
    },
}
fn main() -> color_eyre::Result<()> {
    color_eyre::install()?;
    tracing_subscriber::fmt::init();
    let args = Args::parse();
    let model_info: CLIPModelInfo = toml::from_str(&std::fs::read_to_string(args.model_config)?)?;
    let config: Config = Config {
        db_path: args.db_path,
        gpu_id: args.gpu_id,
        model_info,
    };
    let get_db = || {
        let db: Database = match std::fs::read(config.db_path.join("db.bc")) {
            Ok(fbsrc) => bincode::deserialize(&fbsrc)?,
            _ => Database::default(),
        };
        let db = Arc::new(Mutex::new(db));
        Ok::<_, color_eyre::Report>(db)
    };
    match args.command {
        None => {
            Args::command().print_help()?;
        }
        Some(Commands::Serve) => {
            serve_search(get_db()?, &config)?;
        }
        Some(Commands::UpdateDb {
            media_dirs,
            batch_size,
            tensorrt,
            readers,
        }) => {
            update_db(
                get_db()?,
                &config,
                &media_dirs,
                batch_size,
                tensorrt,
                readers,
            )?;
        }
        Some(Commands::Search { query }) => {
            search_db(get_db()?, &config, query)?;
        }
    }
    Ok(())
}
