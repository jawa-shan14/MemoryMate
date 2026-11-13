"use client";

import {
  useEffect,
  useRef,
  useState,
  type FormEvent,
  type KeyboardEvent,
} from "react";
import WorkspaceSidebar from "@/components/workspace-sidebar";
import {
  getConversationMessages,
  postChatMessage,
  USER_ID,
  type ChatContextRecord,
  type ConversationMessage,
} from "@/lib/memory-api";

type ChatEntry = {
  id: string;
  role: "user" | "assistant";
  content: string;
  createdAt: string;
  memoryContext?: ChatContextRecord[];
  graphContext?: ChatContextRecord[];
};

const HISTORY_KEY = `memorymate:chat:${USER_ID}:conversation`;

function newLocalId() {
  return typeof crypto !== "undefined" && "randomUUID" in crypto
    ? crypto.randomUUID()
    : `${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

function formatTime(value: string) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "";
  return new Intl.DateTimeFormat(undefined, {
    hour: "numeric",
    minute: "2-digit",
  }).format(date);
}

function contextText(value: unknown): string {
  if (typeof value === "string") return value;
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  return "";
}

function relationLabel(record: ChatContextRecord) {
  const source = contextText(record.source);
  const relationship = contextText(record.relationship).replaceAll("_", " ").toLowerCase();
  const target = contextText(record.target);
  if (source || relationship || target) {
    return [source, relationship, target].filter(Boolean).join(" ");
  }
  return Object.entries(record)
    .map(([key, value]) => {
      const text = contextText(value);
      return text ? `${key.replaceAll("_", " ")}: ${text}` : "";
    })
    .filter(Boolean)
    .join(" · ");
}

function MemoryContext({
  memories,
  graph,
}: {
  memories: ChatContextRecord[];
  graph: ChatContextRecord[];
}) {
  return (
    <details className="memory-context">
      <summary>
        <span className="context-summary-icon" aria-hidden="true">✦</span>
        <span>Memory context</span>
        <span className="context-counts">
          {memories.length} {memories.length === 1 ? "memory" : "memories"}
          <i>·</i>
          {graph.length} {graph.length === 1 ? "connection" : "connections"}
        </span>
      </summary>
      <div className="context-content">
        {memories.length === 0 && graph.length === 0 ? (
          <p className="context-empty">No memory or graph context was returned for this response.</p>
        ) : null}
        {memories.length > 0 ? (
          <div className="context-group">
            <h3>Retrieved memories</h3>
            {memories.map((memory, index) => {
              const content = contextText(memory.content);
              const type = contextText(memory.type);
              const confidence = typeof memory.confidence === "number" ? memory.confidence : null;
              return (
                <article className="context-item" key={contextText(memory.id) || `${index}-${content}`}>
                  <div className="context-item-meta">
                    {type ? <span className="context-type">{type}</span> : null}
                    {confidence !== null ? <span>{Math.round(confidence * 100)}% confidence</span> : null}
                  </div>
                  <p>{content || "Memory details were not included in the response."}</p>
                </article>
              );
            })}
          </div>
        ) : null}
        {graph.length > 0 ? (
          <div className="context-group">
            <h3>Retrieved graph context</h3>
            {graph.map((record, index) => (
              <article className="context-item graph-context-item" key={`${index}-${relationLabel(record)}`}>
                <span className="context-node-dot" aria-hidden="true" />
                <p>{relationLabel(record) || "Graph context was returned without a displayable relationship."}</p>
              </article>
            ))}
          </div>
        ) : null}
      </div>
    </details>
  );
}

function historyEntry(message: ConversationMessage): ChatEntry | null {
  if (message.role !== "user" && message.role !== "assistant") return null;
  return {
    id: message.id,
    role: message.role,
    content: message.content,
    createdAt: message.created_at,
  };
}

export default function ChatClient() {
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatEntry[]>([]);
  const [draft, setDraft] = useState("");
  const [initializing, setInitializing] = useState(true);
  const [sending, setSending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [historyError, setHistoryError] = useState<string | null>(null);
  const [storageWarning, setStorageWarning] = useState<string | null>(null);
  const [apiConnected, setApiConnected] = useState<boolean | undefined>(undefined);
  const transcriptRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    let active = true;
    const restoreConversation = async () => {
      let savedConversationId: string | null = null;
      try {
        savedConversationId = window.localStorage.getItem(HISTORY_KEY);
      } catch {
        if (active) setStorageWarning("This browser cannot store the current conversation between visits.");
      }

      if (savedConversationId) {
        if (active) setConversationId(savedConversationId);
        try {
          const history = await getConversationMessages(savedConversationId);
          if (!active) return;
          const entries = history
            .map(historyEntry)
            .filter((entry): entry is ChatEntry => entry !== null);
          setMessages(entries);
          setApiConnected(true);
          setHistoryError(entries.length === 0
            ? "This saved conversation has no messages. Start a new chat to continue."
            : null);
        } catch (loadError) {
          if (active) {
            setApiConnected(false);
            setHistoryError(loadError instanceof Error ? loadError.message : "Could not restore this conversation.");
          }
        }
      }
      if (active) setInitializing(false);
    };

    void restoreConversation();
    return () => { active = false; };
  }, []);

  useEffect(() => {
    if (!messages.length && !sending) return;
    const transcript = transcriptRef.current;
    transcript?.scrollTo({ top: transcript.scrollHeight, behavior: "smooth" });
  }, [messages, sending]);

  const startNewChat = () => {
    if (sending) return;
    setConversationId(null);
    setMessages([]);
    setError(null);
    setHistoryError(null);
    setDraft("");
    try {
      window.localStorage.removeItem(HISTORY_KEY);
    } catch {
      setStorageWarning("This browser could not clear the saved conversation ID.");
    }
    inputRef.current?.focus();
  };

  const submitMessage = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const message = draft.trim();
    if (!message || sending || initializing) return;

    setError(null);
    setHistoryError(null);
    setDraft("");
    setMessages((items) => [...items, {
      id: newLocalId(),
      role: "user",
      content: message,
      createdAt: new Date().toISOString(),
    }]);
    setSending(true);

    try {
      const result = await postChatMessage({
        user_id: USER_ID,
        conversation_id: conversationId,
        message,
        use_memory: true,
        use_graph: true,
      });
      setConversationId(result.conversation_id);
      try {
        window.localStorage.setItem(HISTORY_KEY, result.conversation_id);
        setStorageWarning(null);
      } catch {
        setStorageWarning("This browser could not save the conversation ID. It may not reopen after refresh.");
      }
      setApiConnected(true);
      setMessages((items) => [...items, {
        id: result.message_id,
        role: "assistant",
        content: result.response,
        createdAt: new Date().toISOString(),
        memoryContext: result.retrieved_memories,
        graphContext: result.retrieved_graph_context,
      }]);
    } catch (sendError) {
      setApiConnected(false);
      setError(sendError instanceof Error
        ? sendError.message
        : "MemoryMate could not complete this request.");
    } finally {
      setSending(false);
      inputRef.current?.focus();
    }
  };

  const handleInputKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      event.currentTarget.form?.requestSubmit();
    }
  };

  return (
    <main className="app-shell">
      <WorkspaceSidebar active="chat" connected={apiConnected} />
      <section className="main-column">
        <header className="topbar">
          <div className="breadcrumb"><span>Workspace</span><b>/</b><strong>Chat</strong></div>
          <div className="topbar-right"><span className="avatar">M</span><span className="profile-name">MemoryMate</span></div>
        </header>

        <div className="chat-content">
          <header className="chat-heading">
            <div>
              <div className="eyebrow"><span className="eyebrow-dot" /> YOUR PERSONAL AI</div>
              <h1>Chat with MemoryMate</h1>
              <p>A thoughtful assistant that can draw on your conversations and connected knowledge.</p>
            </div>
            <button type="button" className="new-chat-button" onClick={startNewChat} disabled={sending || initializing}>
              <span aria-hidden="true">+</span> New chat
            </button>
          </header>

          {historyError ? <div className="chat-notice chat-notice-warning" role="status">{historyError}</div> : null}
          {storageWarning ? <div className="chat-notice chat-notice-warning" role="status">{storageWarning}</div> : null}

          <section className="chat-surface" aria-label="Chat conversation">
            <div className="chat-transcript" ref={transcriptRef} role="log" aria-live="polite" aria-relevant="additions text">
              {initializing ? (
                <div className="chat-state"><span className="spinner" />Restoring your conversation…</div>
              ) : null}
              {!initializing && messages.length === 0 ? (
                <div className="chat-empty">
                  <div className="chat-empty-mark" aria-hidden="true">✦</div>
                  <h2>What would you like to talk about?</h2>
                  <p>Share a thought, ask a question, or pick up where your memory can help.</p>
                  <span>Send a message to begin a conversation.</span>
                </div>
              ) : null}
              {messages.map((message) => (
                <article className={`chat-message ${message.role === "user" ? "chat-message-user" : "chat-message-assistant"}`} key={message.id}>
                  <div className={`message-avatar ${message.role === "user" ? "user-avatar" : "assistant-avatar"}`}>
                    {message.role === "user" ? "Y" : "✦"}
                  </div>
                  <div className="message-body">
                    <div className="message-byline">
                      <strong>{message.role === "user" ? "You" : "MemoryMate"}</strong>
                      <time>{formatTime(message.createdAt)}</time>
                    </div>
                    <p className="message-text">{message.content}</p>
                    {message.role === "assistant" && message.memoryContext && message.graphContext ? (
                      <MemoryContext memories={message.memoryContext} graph={message.graphContext} />
                    ) : null}
                  </div>
                </article>
              ))}
              {sending ? (
                <div className="chat-message chat-message-assistant" role="status">
                  <div className="message-avatar assistant-avatar">✦</div>
                  <div className="message-body">
                    <div className="message-byline"><strong>MemoryMate</strong><span>Thinking</span></div>
                    <div className="typing-indicator" aria-label="MemoryMate is generating a response"><i /><i /><i /></div>
                  </div>
                </div>
              ) : null}
            </div>

            {error ? (
              <div className="chat-notice chat-notice-error" role="alert">
                <strong>Message not completed</strong>
                <span>{error}</span>
                <small>The backend did not return a response. Check your conversation before resending to avoid duplicates.</small>
              </div>
            ) : null}

            <form className="chat-composer" onSubmit={(event) => void submitMessage(event)}>
              <label className="sr-only" htmlFor="chat-input">Message MemoryMate</label>
              <textarea
                id="chat-input"
                ref={inputRef}
                value={draft}
                onChange={(event) => setDraft(event.target.value)}
                onKeyDown={handleInputKeyDown}
                placeholder="Message MemoryMate…"
                maxLength={10000}
                rows={2}
                disabled={initializing || sending}
              />
              <div className="composer-footer">
                <span>Enter to send <i>·</i> Shift + Enter for a new line</span>
                <div>
                  {draft.length > 0 ? <span className="character-count">{draft.length.toLocaleString()} / 10,000</span> : null}
                  <button type="submit" className="send-button" disabled={initializing || sending || !draft.trim()} aria-label="Send message">
                    {sending ? <span className="send-spinner" /> : (
                      <svg aria-hidden="true" width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M5 12h14M13 6l6 6-6 6" />
                      </svg>
                    )}
                  </button>
                </div>
              </div>
            </form>
            <p className="chat-disclaimer">MemoryMate may make mistakes. Check important information against its source.</p>
          </section>
          <footer className="page-footer"><span>MemoryMate</span><span>Context shown is what the backend returned for each response.</span></footer>
        </div>
      </section>
    </main>
  );
}
