use std::sync::Arc;

use aionui_ai_agent::{
    AgentStreamEvent,
    protocol::events::{FinishEventData, TipType, TipsEventData},
};
use aionui_common::now_ms;
use aionui_conversation::stream_relay::StreamRelay;
use aionui_db::{
    IConversationRepository, MessagePageDirection, MessagePageParams, SqliteConversationRepository,
    init_database_memory, models::ConversationRow,
};
use aionui_realtime::BroadcastEventBus;
use serde_json::json;
use tokio::sync::broadcast;

async fn setup_repo() -> (Arc<SqliteConversationRepository>, aionui_db::Database) {
    let db = init_database_memory().await.unwrap();
    let repo = Arc::new(SqliteConversationRepository::new(db.pool().clone()));
    let now = now_ms();
    repo.create(&ConversationRow {
        id: "conv-1".into(),
        user_id: "system_default_user".into(),
        name: "Stream relay tips test".into(),
        r#type: "acp".into(),
        extra: "{}".into(),
        model: None,
        status: Some("running".into()),
        source: Some("aionui".into()),
        channel_chat_id: None,
        pinned: false,
        pinned_at: None,
        created_at: now,
        updated_at: now,
        project_id: None,
        folder_id: None,
        name_source: None,
    })
    .await
    .unwrap();

    (repo, db)
}

#[tokio::test]
async fn persist_info_tip_preserves_code_and_params() {
    let (repo, _db) = setup_repo().await;
    let bus = Arc::new(BroadcastEventBus::new(64));
    let (tx, _) = broadcast::channel(64);

    let relay = StreamRelay::new(
        "conv-1".into(),
        "asst-1".into(),
        "turn-1".into(),
        "system_default_user".into(),
        repo.clone(),
        bus,
    );

    let rx = tx.subscribe();
    tx.send(AgentStreamEvent::Tips(TipsEventData {
        content: String::new(),
        tip_type: TipType::Info,
        code: Some("ACP_EMPTY_TURN".into()),
        params: Some(json!({ "scope": "session", "cleared": 12 })),
        supersedes_key: None,
    }))
    .unwrap();
    tx.send(AgentStreamEvent::Finish(FinishEventData::default())).unwrap();

    relay.consume(rx).await;

    let messages = repo
        .list_messages_page(
            "system_default_user",
            "conv-1",
            &MessagePageParams {
                limit: 100,
                direction: MessagePageDirection::InitialLatest,
            },
        )
        .await
        .unwrap();
    let tip = messages
        .items
        .iter()
        .find(|row| row.r#type == "tips")
        .expect("info tip should be persisted");

    assert_eq!(tip.status.as_deref(), Some("finish"));

    let content: serde_json::Value = serde_json::from_str(&tip.content).unwrap();
    assert_eq!(content["content"], "");
    assert_eq!(content["type"], "info");
    assert_eq!(content["code"], "ACP_EMPTY_TURN");
    assert_eq!(content["params"], json!({ "scope": "session", "cleared": 12 }));
}

/// A retry tip is the same card counting up, so its merge key has to survive
/// persistence: reloading the conversation folds history with the same rule the
/// live stream uses, and without the key a stalled turn comes back as a stack of
/// near-identical "Reconnecting... N/5" cards.
#[tokio::test]
async fn persist_warning_tip_preserves_supersedes_key() {
    let (repo, _db) = setup_repo().await;
    let bus = Arc::new(BroadcastEventBus::new(64));
    let (tx, _) = broadcast::channel(64);

    let relay = StreamRelay::new(
        "conv-1".into(),
        "asst-1".into(),
        "turn-1".into(),
        "system_default_user".into(),
        repo.clone(),
        bus,
    );

    let rx = tx.subscribe();
    for attempt in 1..=3 {
        tx.send(AgentStreamEvent::Tips(TipsEventData {
            content: format!("Reconnecting... {attempt}/5"),
            tip_type: TipType::Warning,
            code: None,
            params: None,
            supersedes_key: Some("codex-retry:turn-1".into()),
        }))
        .unwrap();
    }
    tx.send(AgentStreamEvent::Finish(FinishEventData::default())).unwrap();

    relay.consume(rx).await;

    let messages = repo
        .list_messages_page(
            "system_default_user",
            "conv-1",
            &MessagePageParams {
                limit: 100,
                direction: MessagePageDirection::InitialLatest,
            },
        )
        .await
        .unwrap();
    let tips: Vec<_> = messages.items.iter().filter(|row| row.r#type == "tips").collect();
    assert_eq!(tips.len(), 3, "every attempt is persisted; the merge happens on read");

    for row in tips {
        let content: serde_json::Value = serde_json::from_str(&row.content).unwrap();
        assert_eq!(content["supersedes_key"], "codex-retry:turn-1");
    }
}
