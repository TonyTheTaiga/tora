use fred::prelude::*;
use sqlx::{self, PgPool};
use std::fmt;
use std::time::Duration;

use crate::{state::AppState, types::OutLog};

const LOOP_TIMER: Duration = Duration::from_secs(15);

#[derive(Debug)]
struct PublishError {
    log_id: i64,
    source: Box<dyn std::error::Error + Send + Sync>,
}
impl fmt::Display for PublishError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "publish failed for log {}", self.log_id)
    }
}
impl std::error::Error for PublishError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&*self.source)
    }
}

async fn get_unpublished_rows(db_pool: &PgPool) -> Vec<OutLog> {
    sqlx::query_as::<_, OutLog>(
        r#"
        with ranked as (
            select
                lo.*,
                row_number() over (partition by lo.experiment_id order by lo.id asc) as rn
            from public.log_outbox lo
            where lo.processed_at is null
        )
        select 
            id,
            experiment_id::text,
            msg_id::text,
            created_at,
            payload,
            attempt_count,
            next_attempt_at,
            processed_at 
        from ranked 
        where rn = 1 and next_attempt_at <= now()
        "#,
    )
    .fetch_all(db_pool)
    .await
    .expect("failed to fetch unpublished rows")
}

async fn publish_outlog(
    publisher_client: &Client,
    log: OutLog,
) -> std::result::Result<i64, PublishError> {
    let experiment_id = log.experiment_id;
    let channel = format!("log:exp:{experiment_id}");
    let payload = Value::String(log.payload.to_string().into());
    let _: i64 = publisher_client
        .publish(channel, payload)
        .await
        .map_err(|e| PublishError {
            log_id: log.id,
            source: e.into(),
        })?;

    Ok(log.id)
}

async fn report_published_logs(
    db_pool: &PgPool,
    log_ids: &[i64],
) -> Result<(), Box<dyn std::error::Error>> {
    if log_ids.is_empty() {
        return Ok(());
    }

    sqlx::query(
        r#"
        UPDATE public.log_outbox
        SET processed_at = now()
        WHERE id = ANY($1)
        "#,
    )
    .bind(log_ids)
    .execute(db_pool)
    .await?;

    Ok(())
}

async fn report_failed_logs(
    db_pool: &PgPool,
    log_ids: &[i64],
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    if log_ids.is_empty() {
        return Ok(());
    }

    // Increment attempt_count and schedule the next attempt with a simple linear backoff.
    // Backoff: (attempt_count + 1) * 30 seconds
    sqlx::query(
        r#"
        UPDATE public.log_outbox
        SET attempt_count = attempt_count + 1,
            next_attempt_at = now() + ((attempt_count + 1) * interval '30 seconds')
        WHERE id = ANY($1)
        "#,
    )
    .bind(log_ids)
    .execute(db_pool)
    .await?;

    Ok(())
}

pub async fn run_worker(state: AppState) {
    loop {
        let rows = get_unpublished_rows(&state.db_pool).await;
        println!("got {:?} rows", rows.len());

        let client = state.vk_pool.next_connected();
        let mut published_ids: Vec<i64> = vec![];
        let mut failed_ids: Vec<i64> = vec![];
        for log in rows {
            match publish_outlog(client, log).await {
                Ok(log_id) => published_ids.push(log_id),
                Err(e) => failed_ids.push(e.log_id),
            }
        }

        if !published_ids.is_empty() {
            if let Err(e) = report_published_logs(&state.db_pool, &published_ids).await {
                eprintln!("failed to mark published logs as processed: {e}");
            }
        }

        if !failed_ids.is_empty() {
            if let Err(e) = report_failed_logs(&state.db_pool, &failed_ids).await {
                eprintln!("failed to mark failed logs: {e}");
            }
        }

        tokio::time::sleep(LOOP_TIMER).await;
    }
}
