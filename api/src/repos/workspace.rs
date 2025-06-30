use serde::Deserialize;

#[derive(Deserialize)]
struct Workspace {
    name: String,
}

pub async fn list_workspaces() -> String {
    "list workspaces".to_string()
}

