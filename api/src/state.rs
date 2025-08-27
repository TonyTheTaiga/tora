use crate::settings::Settings;
use fred::clients::Client;
use sqlx::PgPool;

#[derive(Clone)]
pub struct AppState {
    pub db_pool: PgPool,
    pub settings: Settings,
    pub vk_client: Client,
}
