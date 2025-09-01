use std::env;

#[derive(Debug, Clone)]
pub struct Settings {
    pub frontend_url: String,
    pub database_url: String,
    pub supabase_url: String,
    pub supabase_api_key: String,
    pub supabase_jwt_secret: String,
    pub redis_url: String,
    pub outbox_polling_interval: u64,
}

fn load_env_value(env_name: String) -> Result<String, Box<dyn std::error::Error>> {
    match env::var(&env_name) {
        Ok(env_value) => Ok(env_value),
        Err(_) => {
            eprintln!("Error: Couldn't load env var {env_name}");
            Err(format!("{env_name} not set!").into())
        }
    }
}

impl Settings {
    pub fn from_env() -> Settings {
        let database_url =
            load_env_value("DATABASE_URL".to_string()).expect("DATABASE_URL not set!");

        let frontend_url =
            load_env_value("FRONTEND_URL".to_string()).expect("FRONTEND_URL not set!");

        let supabase_url =
            load_env_value("SUPABASE_URL".to_string()).expect("SUPABASE_URL not set!");

        let supabase_api_key =
            load_env_value("SUPABASE_API_KEY".to_string()).expect("SUPABASE_API_KEY not set!");

        let supabase_jwt_secret = load_env_value("SUPABASE_JWT_SECRET".to_string())
            .expect("SUPABASE_JWT_SECRET not set!");

        let redis_url = load_env_value("REDIS_URL".to_string()).expect("REDIS_URL not set!");
        let outbox_polling_interval: u64 = match load_env_value("OUTBOX_POLLING_INTERVAL".into()) {
            Ok(s) => s
                .parse::<u64>()
                .expect("OUTBOX_POLLING_INTERVAL must be an integer"),
            Err(_) => 5,
        };

        Settings {
            frontend_url,
            database_url,
            supabase_url,
            supabase_api_key,
            supabase_jwt_secret,
            redis_url,
            outbox_polling_interval,
        }
    }
}
