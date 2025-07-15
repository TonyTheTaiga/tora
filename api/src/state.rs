use crate::settings::Settings;

#[derive(Clone)]
pub struct AppState {
    pub db_pool: sqlx::PgPool,
    pub settings: Settings,
}
