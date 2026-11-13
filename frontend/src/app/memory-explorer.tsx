"use client";

import {
  useEffect,
  useMemo,
  useRef,
  useState,
  type KeyboardEvent,
  type ReactNode,
} from "react";
import WorkspaceSidebar from "@/components/workspace-sidebar";
import {
  deleteMemory,
  getConversationMessages,
  getConversations,
  getKnowledgeGraph,
  getMemories,
  getMemoryStats,
  searchMemories,
  updateMemory,
  type Conversation,
  type ConversationMessage,
  type GraphEdge,
  type GraphNode,
  type Memory,
  type MemoryStats,
  type MemoryType,
} from "@/lib/memory-api";

const MEMORY_GROUPS: { type: MemoryType; label: string; tone: string }[] = [
  { type: "episodic", label: "Episodic", tone: "violet" },
  { type: "semantic", label: "Semantic", tone: "teal" },
  { type: "consolidated", label: "Consolidated", tone: "amber" },
];

function formatDate(value: string) {
  return new Intl.DateTimeFormat(undefined, {
    month: "short",
    day: "numeric",
    year: "numeric",
  }).format(new Date(value));
}

function Icon({
  name,
  size = 18,
}: {
  name: "search" | "spark" | "network" | "clock" | "edit" | "trash" | "close" | "plus" | "minus" | "arrow";
  size?: number;
}) {
  const paths: Record<typeof name, ReactNode> = {
    search: <><circle cx="11" cy="11" r="7" /><path d="m20 20-4-4" /></>,
    spark: <><path d="m12 3 1.9 5.8L20 11l-6.1 2.2L12 19l-1.9-5.8L4 11l6.1-2.2L12 3Z" /><path d="m19 14 1 2.5 2.5 1-2.5 1L19 21l-1-2.5-2.5-1 2.5-1L19 14Z" /></>,
    network: <><circle cx="12" cy="5" r="2" /><circle cx="5" cy="18" r="2" /><circle cx="19" cy="18" r="2" /><path d="m11 7-5 9m7-9 5 9M7 18h10" /></>,
    clock: <><circle cx="12" cy="12" r="9" /><path d="M12 7v5l3 2" /></>,
    edit: <><path d="m15 5 4 4M4 20l4-.8L19 8a2.1 2.1 0 0 0-3-3L5 16l-1 4Z" /></>,
    trash: <><path d="M4 7h16M10 11v6m4-6v6M6 7l1 13h10l1-13M9 7V4h6v3" /></>,
    close: <><path d="m6 6 12 12M18 6 6 18" /></>,
    plus: <><path d="M12 5v14M5 12h14" /></>,
    minus: <path d="M5 12h14" />,
    arrow: <><path d="M5 12h14M13 6l6 6-6 6" /></>,
  };
  return (
    <svg
      aria-hidden="true"
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.7"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      {paths[name]}
    </svg>
  );
}

function ErrorNotice({ children }: { children: ReactNode }) {
  return <div className="notice notice-error" role="alert">{children}</div>;
}

function GraphCanvas({
  nodes,
  edges,
  query,
  onSelect,
  selected,
}: {
  nodes: GraphNode[];
  edges: GraphEdge[];
  query: string;
  onSelect: (node: GraphNode) => void;
  selected: GraphNode | null;
}) {
  const [zoom, setZoom] = useState(1);
  const filteredNodes = useMemo(() => {
    const term = query.trim().toLocaleLowerCase();
    return term
      ? nodes.filter((node) => node.label.toLocaleLowerCase().includes(term))
      : nodes;
  }, [nodes, query]);
  const visibleIds = new Set(filteredNodes.map((node) => node.id));
  const visibleEdges = edges.filter(
    (edge) => visibleIds.has(edge.source) && visibleIds.has(edge.target),
  );
  const positions = new Map(
    filteredNodes.map((node, index) => {
      const angle = (2 * Math.PI * index) / Math.max(filteredNodes.length, 1) - Math.PI / 2;
      const radius = filteredNodes.length === 1 ? 0 : Math.min(154, 44 + filteredNodes.length * 8);
      return [
        node.id,
        { x: 380 + Math.cos(angle) * radius, y: 198 + Math.sin(angle) * radius },
      ];
    }),
  );

  if (nodes.length === 0) {
    return (
      <div className="graph-empty">
        <span className="empty-icon"><Icon name="network" size={22} /></span>
        <strong>No entities yet</strong>
        <p>Entities and relationships appear here as your conversations build context.</p>
      </div>
    );
  }
  if (filteredNodes.length === 0) {
    return (
      <div className="graph-empty">
        <span className="empty-icon"><Icon name="search" size={22} /></span>
        <strong>No matching entities</strong>
        <p>Try a different search to explore another part of your graph.</p>
      </div>
    );
  }

  return (
    <div className="graph-stage">
      <svg
        className="graph-svg"
        viewBox="0 0 760 400"
        role="img"
        aria-label={`${filteredNodes.length} knowledge graph entities and ${visibleEdges.length} relationships`}
      >
        <defs>
          <marker id="edge-arrow" markerWidth="7" markerHeight="7" refX="5" refY="3.5" orient="auto">
            <path d="M0 0 6 3.5 0 7" fill="none" stroke="#a9a8b8" strokeWidth="1.2" />
          </marker>
        </defs>
        <g transform={`translate(380 200) scale(${zoom}) translate(-380 -200)`}>
          {visibleEdges.map((edge, index) => {
            const from = positions.get(edge.source);
            const to = positions.get(edge.target);
            if (!from || !to) return null;
            const centerX = (from.x + to.x) / 2;
            const centerY = (from.y + to.y) / 2;
            return (
              <g key={`${edge.source}-${edge.target}-${edge.label}-${index}`}>
                <line
                  x1={from.x}
                  y1={from.y}
                  x2={to.x}
                  y2={to.y}
                  className="graph-edge"
                  markerEnd="url(#edge-arrow)"
                />
                <text x={centerX} y={centerY - 5} className="edge-label">{edge.label.replaceAll("_", " ")}</text>
              </g>
            );
          })}
          {filteredNodes.map((node) => {
            const point = positions.get(node.id);
            if (!point) return null;
            const isSelected = selected?.id === node.id;
            return (
              <g
                key={node.id}
                className={`graph-node${isSelected ? " is-selected" : ""}`}
                role="button"
                tabIndex={0}
                aria-label={`Select ${node.label}, ${node.type}`}
                onClick={() => onSelect(node)}
                onKeyDown={(event: KeyboardEvent<SVGGElement>) => {
                  if (event.key === "Enter" || event.key === " ") {
                    event.preventDefault();
                    onSelect(node);
                  }
                }}
              >
                <circle cx={point.x} cy={point.y} r={isSelected ? 11 : 8} />
                <text x={point.x} y={point.y + 25} textAnchor="middle">{node.label}</text>
              </g>
            );
          })}
        </g>
      </svg>
      <div className="graph-zoom" aria-label="Graph zoom controls">
        <button type="button" aria-label="Zoom in" onClick={() => setZoom((value) => Math.min(1.6, value + 0.15))}>
          <Icon name="plus" size={16} />
        </button>
        <button type="button" aria-label="Zoom out" onClick={() => setZoom((value) => Math.max(0.7, value - 0.15))}>
          <Icon name="minus" size={16} />
        </button>
      </div>
      <div className="graph-legend"><span /> Entity <i /> Relationship</div>
    </div>
  );
}

function ConversationDialog({
  conversation,
  messages,
  loading,
  error,
  onClose,
}: {
  conversation: Conversation | null;
  messages: ConversationMessage[];
  loading: boolean;
  error: string | null;
  onClose: () => void;
}) {
  if (!conversation) return null;
  return (
    <div className="dialog-backdrop" onMouseDown={(event) => {
      if (event.target === event.currentTarget) onClose();
    }}>
      <section className="conversation-dialog" role="dialog" aria-modal="true" aria-labelledby="conversation-title">
        <div className="dialog-head">
          <div>
            <span className="eyebrow">SOURCE CONVERSATION</span>
            <h2 id="conversation-title">{conversation.title || "Untitled conversation"}</h2>
          </div>
          <button type="button" className="icon-button" aria-label="Close conversation" onClick={onClose}>
            <Icon name="close" />
          </button>
        </div>
        {loading ? <div className="dialog-state">Loading conversation…</div> : null}
        {error ? <ErrorNotice>{error}</ErrorNotice> : null}
        {!loading && !error && messages.length === 0 ? (
          <div className="dialog-state">No messages were found in this conversation.</div>
        ) : null}
        <div className="message-list">
          {messages.map((message) => (
            <article key={message.id} className={`conversation-message ${message.role === "user" ? "from-user" : ""}`}>
              <span>{message.role === "user" ? "You" : "MemoryMate"}</span>
              <p>{message.content}</p>
              <time>{formatDate(message.created_at)}</time>
            </article>
          ))}
        </div>
      </section>
    </div>
  );
}

export default function MemoryExplorer() {
  const [memories, setMemories] = useState<Memory[]>([]);
  const [stats, setStats] = useState<MemoryStats | null>(null);
  const [nodes, setNodes] = useState<GraphNode[]>([]);
  const [edges, setEdges] = useState<GraphEdge[]>([]);
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [memoryLoading, setMemoryLoading] = useState(true);
  const [graphLoading, setGraphLoading] = useState(true);
  const [memoryError, setMemoryError] = useState<string | null>(null);
  const [statsError, setStatsError] = useState<string | null>(null);
  const [graphError, setGraphError] = useState<string | null>(null);
  const [conversationError, setConversationError] = useState<string | null>(null);
  const [search, setSearch] = useState("");
  const [searchResult, setSearchResult] = useState<{ query: string; ids: Set<string> } | null>(null);
  const [searchError, setSearchError] = useState<string | null>(null);
  const [activeFilter, setActiveFilter] = useState<MemoryType | "all">("all");
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editContent, setEditContent] = useState("");
  const [savingId, setSavingId] = useState<string | null>(null);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null);
  const [conversation, setConversation] = useState<Conversation | null>(null);
  const [conversationMessages, setConversationMessages] = useState<ConversationMessage[]>([]);
  const [messagesLoading, setMessagesLoading] = useState(false);
  const [messagesError, setMessagesError] = useState<string | null>(null);
  const [apiConnected, setApiConnected] = useState(false);
  const searchSequence = useRef(0);

  useEffect(() => {
    let cancelled = false;
    getMemories().then((items) => {
      if (!cancelled) {
        setMemories(items);
        setApiConnected(true);
      }
    }).catch((error: unknown) => {
      if (!cancelled) {
        setMemoryError(error instanceof Error ? error.message : "Could not load memories.");
        setApiConnected(false);
      }
    }).finally(() => {
      if (!cancelled) setMemoryLoading(false);
    });
    getMemoryStats().then((value) => {
      if (!cancelled) setStats(value);
    }).catch((error: unknown) => {
      if (!cancelled) setStatsError(error instanceof Error ? error.message : "Could not load memory statistics.");
    });
    getKnowledgeGraph().then((graph) => {
      if (cancelled) return;
      if (!graph.available) {
        setGraphError("The knowledge graph is currently unavailable.");
        return;
      }
      setNodes(graph.nodes);
      setEdges(graph.edges);
    }).catch((error: unknown) => {
      if (!cancelled) setGraphError(error instanceof Error ? error.message : "Could not load the knowledge graph.");
    }).finally(() => {
      if (!cancelled) setGraphLoading(false);
    });
    getConversations().then((items) => {
      if (!cancelled) setConversations(items);
    }).catch((error: unknown) => {
      if (!cancelled) setConversationError(error instanceof Error ? error.message : "Could not load conversations.");
    });
    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    const query = search.trim();
    const sequence = ++searchSequence.current;
    if (!query) return;
    const timer = window.setTimeout(() => {
      searchMemories(query).then((hits) => {
        if (sequence === searchSequence.current) {
          setSearchResult({ query, ids: new Set(hits.map((hit) => hit.id)) });
        }
      }).catch((error: unknown) => {
        if (sequence === searchSequence.current) {
          setSearchError(error instanceof Error ? error.message : "Memory search failed.");
        }
      });
    }, 300);
    return () => window.clearTimeout(timer);
  }, [search]);

  const searchQuery = search.trim();
  const searchLoading = Boolean(searchQuery && searchResult?.query !== searchQuery && !searchError);
  const visibleMemories = useMemo(() => {
    const searchIds = searchQuery
      ? searchResult?.query === searchQuery ? searchResult.ids : new Set<string>()
      : null;
    return memories.filter((memory) => {
      const typeMatches = activeFilter === "all" || memory.memory_type === activeFilter;
      const searchMatches = searchIds === null || searchIds.has(memory.id);
      return typeMatches && searchMatches;
    });
  }, [activeFilter, memories, searchQuery, searchResult]);

  const openConversation = async (id: string) => {
    const found = conversations.find((item) => item.id === id);
    const source = found ?? {
      id,
      title: `Conversation ${id.slice(0, 8)}`,
      summary: null,
      created_at: "",
      updated_at: "",
      message_count: 0,
    };
    setConversation(source);
    setConversationMessages([]);
    setMessagesError(null);
    setMessagesLoading(true);
    try {
      setConversationMessages(await getConversationMessages(id));
    } catch (error) {
      setMessagesError(error instanceof Error ? error.message : "Could not load this conversation.");
    } finally {
      setMessagesLoading(false);
    }
  };

  const saveEdit = async (memory: Memory) => {
    const content = editContent.trim();
    if (!content) {
      setActionError("Memory content cannot be empty.");
      return;
    }
    setSavingId(memory.id);
    setActionError(null);
    try {
      const updated = await updateMemory(memory.id, content);
      setMemories((items) => items.map((item) => item.id === updated.id ? updated : item));
      setEditingId(null);
    } catch (error) {
      setActionError(error instanceof Error ? error.message : "Could not update this memory.");
    } finally {
      setSavingId(null);
    }
  };

  const removeMemory = async (memory: Memory) => {
    if (!window.confirm("Delete this memory? This will remove it from your active memory list.")) return;
    setDeletingId(memory.id);
    setActionError(null);
    try {
      await deleteMemory(memory.id);
      setMemories((items) => items.filter((item) => item.id !== memory.id));
      setStats((current) => current ? {
        ...current,
        total_memories: Math.max(0, current.total_memories - 1),
        episodic_count: current.episodic_count - Number(memory.memory_type === "episodic"),
        semantic_count: current.semantic_count - Number(memory.memory_type === "semantic"),
        consolidated_count: current.consolidated_count - Number(memory.memory_type === "consolidated"),
      } : current);
    } catch (error) {
      setActionError(error instanceof Error ? error.message : "Could not delete this memory.");
    } finally {
      setDeletingId(null);
    }
  };

  const startEdit = (memory: Memory) => {
    setActionError(null);
    setEditingId(memory.id);
    setEditContent(memory.content);
  };

  const conversationTitles = useMemo(
    () => new Map(conversations.map((item) => [item.id, item.title || "Untitled conversation"])),
    [conversations],
  );

  return (
    <main className="app-shell">
      <WorkspaceSidebar active="explorer" connected={apiConnected} />

      <section className="main-column">
        <header className="topbar">
          <div className="breadcrumb"><span>Workspace</span><b>/</b><strong>Memory explorer</strong></div>
          <div className="topbar-right"><span className="avatar">M</span><span className="profile-name">My memory</span></div>
        </header>

        <div className="content-wrap">
          <section className="intro">
            <div className="intro-copy">
              <div className="eyebrow"><span className="eyebrow-dot" /> YOUR SECOND BRAIN</div>
              <h1>Memory, <span>in focus.</span></h1>
              <p>Explore the ideas, experiences, and connections your AI has remembered.</p>
            </div>
            <label className="search-box">
              <Icon name="search" size={19} />
              <input
                aria-label="Search memories"
                placeholder="Search your memories"
                value={search}
                onChange={(event) => {
                  setSearch(event.target.value);
                  setSearchError(null);
                }}
              />
              {search ? <button type="button" className="clear-search" aria-label="Clear search" onClick={() => setSearch("")}><Icon name="close" size={16} /></button> : null}
            </label>
          </section>

          <section className="stats-row" aria-label="Memory overview">
            <article className="stat-card stat-total">
              <span className="stat-symbol"><Icon name="spark" size={17} /></span>
              <div><span className="stat-label">Total memories</span><strong>{memoryLoading || !stats ? "—" : stats.total_memories}</strong></div>
              <span className="stat-foot">Across every layer</span>
            </article>
            {MEMORY_GROUPS.slice(0, 2).map((group) => (
              <article className="stat-card" key={group.type}>
                <span className={`stat-dot ${group.tone}`} />
                <div><span className="stat-label">{group.label}</span><strong>{memoryLoading || !stats ? "—" : group.type === "episodic" ? stats.episodic_count : stats.semantic_count}</strong></div>
                <span className="stat-foot">{group.type === "episodic" ? "Moments & experiences" : "Facts & preferences"}</span>
              </article>
            ))}
            <article className="stat-card">
              <span className="stat-symbol graph-symbol"><Icon name="network" size={17} /></span>
              <div><span className="stat-label">Graph entities</span><strong>{graphLoading ? "—" : nodes.length}</strong></div>
              <span className="stat-foot">{edges.length} relationships</span>
            </article>
          </section>

          <div className="section-heading memories-heading" id="memories">
            <div><div className="eyebrow">YOUR MEMORY LIBRARY</div><h2>Memories</h2></div>
            <div className="filter-pills" aria-label="Filter memories">
              {(["all", "episodic", "semantic", "consolidated"] as const).map((filter) => (
                <button
                  type="button"
                  key={filter}
                  className={`filter-pill${activeFilter === filter ? " selected" : ""}`}
                  aria-pressed={activeFilter === filter}
                  onClick={() => setActiveFilter(filter)}
                >
                  {filter === "all" ? "All" : filter[0].toUpperCase() + filter.slice(1)}
                </button>
              ))}
            </div>
          </div>
          {memoryError ? <ErrorNotice>{memoryError}</ErrorNotice> : null}
          {statsError ? <ErrorNotice>{statsError}</ErrorNotice> : null}
          {searchError ? <ErrorNotice>Search failed: {searchError}</ErrorNotice> : null}
          {actionError ? <ErrorNotice>{actionError}</ErrorNotice> : null}
          {conversationError ? <div className="inline-warning">Conversation titles could not be loaded: {conversationError}</div> : null}
          <section className="memory-library" aria-label="Memory library">
            {memoryLoading || searchLoading ? (
              <div className="loading-state"><span className="spinner" />{memoryLoading ? "Loading your memories…" : "Searching your memories…"}</div>
            ) : null}
            {!memoryLoading && !memoryError && !searchLoading && visibleMemories.length === 0 ? (
              <div className="empty-library">
                <span className="empty-icon"><Icon name={search ? "search" : "spark"} size={22} /></span>
                <strong>{search ? "No memories found" : "Your memory is taking shape"}</strong>
                <p>{search ? "Try another phrase or clear your search." : "Memories from your conversations will appear here."}</p>
              </div>
            ) : null}
            {!memoryLoading && !searchLoading ? MEMORY_GROUPS
              .filter((group) => activeFilter === "all" || activeFilter === group.type)
              .map((group) => {
                const items = visibleMemories.filter((memory) => memory.memory_type === group.type);
                if (items.length === 0) return null;
                return (
                  <div className="memory-group" key={group.type}>
                    <div className="group-heading">
                      <span className={`group-marker ${group.tone}`} />
                      <h3>{group.label}</h3>
                      <span className="group-count">{items.length}</span>
                    </div>
                    <div className="memory-grid">
                      {items.map((memory) => (
                        <article className="memory-card" key={memory.id}>
                          <div className="memory-card-top">
                            <span className={`memory-type ${group.tone}`}>{group.label}</span>
                            <div className="memory-actions">
                              <button type="button" className="icon-button" aria-label={`Edit memory: ${memory.content.slice(0, 50)}`} onClick={() => startEdit(memory)} disabled={editingId === memory.id || deletingId === memory.id}>
                                <Icon name="edit" size={16} />
                              </button>
                              <button type="button" className="icon-button danger-hover" aria-label={`Delete memory: ${memory.content.slice(0, 50)}`} onClick={() => void removeMemory(memory)} disabled={deletingId === memory.id}>
                                {deletingId === memory.id ? <span className="mini-spinner" /> : <Icon name="trash" size={16} />}
                              </button>
                            </div>
                          </div>
                          {editingId === memory.id ? (
                            <div className="edit-form">
                              <textarea aria-label="Edit memory content" value={editContent} onChange={(event) => setEditContent(event.target.value)} />
                              <div className="edit-actions">
                                <button type="button" className="button-quiet" onClick={() => setEditingId(null)} disabled={savingId === memory.id}>Cancel</button>
                                <button type="button" className="button-primary" onClick={() => void saveEdit(memory)} disabled={savingId === memory.id}>
                                  {savingId === memory.id ? "Saving…" : "Save changes"}
                                </button>
                              </div>
                            </div>
                          ) : <p className="memory-content">{memory.content}</p>}
                          {memory.tags.length > 0 ? <div className="tag-row">{memory.tags.map((tag) => <span className="memory-tag" key={tag}>{tag}</span>)}</div> : null}
                          <div className="memory-card-footer">
                            <span><Icon name="clock" size={14} />{formatDate(memory.created_at)}</span>
                            {memory.source_conversation_id ? (
                              <button type="button" className="source-link" onClick={() => void openConversation(memory.source_conversation_id!)}>
                                <span className="source-icon">↗</span>
                                {conversationTitles.get(memory.source_conversation_id) ?? `Conversation ${memory.source_conversation_id.slice(0, 8)}`}
                              </button>
                            ) : <span className="source-missing">No source conversation</span>}
                          </div>
                        </article>
                      ))}
                    </div>
                  </div>
                );
              }) : null}
          </section>

          <section className="graph-section" id="knowledge-graph">
            <div className="section-heading graph-heading">
              <div><div className="eyebrow">CONNECTED KNOWLEDGE</div><h2>Knowledge graph</h2></div>
              <div className="graph-count"><strong>{nodes.length}</strong> entities <span>·</span> <strong>{edges.length}</strong> relationships</div>
            </div>
            <div className="graph-card">
              {graphError ? <div className="graph-error"><ErrorNotice>{graphError}</ErrorNotice></div> : null}
              {graphLoading ? <div className="loading-state graph-loading"><span className="spinner" />Connecting to your knowledge graph…</div> : null}
              {!graphLoading && !graphError ? (
                <>
                  <GraphCanvas nodes={nodes} edges={edges} query={search} selected={selectedNode} onSelect={setSelectedNode} />
                  {selectedNode ? (
                    <div className="entity-detail" aria-live="polite">
                      <span className="entity-detail-dot" />
                      <div><strong>{selectedNode.label}</strong><span>{selectedNode.type}</span></div>
                      <button type="button" className="icon-button" aria-label="Clear selected entity" onClick={() => setSelectedNode(null)}><Icon name="close" size={16} /></button>
                    </div>
                  ) : null}
                  <div className="relationship-list">
                    <div className="relationship-title"><span>RELATIONSHIPS</span><span>{edges.length}</span></div>
                    {edges.length > 0 ? edges.slice(0, 8).map((edge, index) => {
                      const source = nodes.find((node) => node.id === edge.source);
                      const target = nodes.find((node) => node.id === edge.target);
                      if (!source || !target) return null;
                      return <div className="relationship-row" key={`${edge.source}-${edge.target}-${edge.label}-${index}`}>
                        <span className="relationship-entity">{source.label}</span><span className="relationship-edge">{edge.label.replaceAll("_", " ")}</span><span className="relationship-entity">{target.label}</span>
                      </div>;
                    }) : <p className="relationship-empty">No relationships are available yet.</p>}
                  </div>
                </>
              ) : null}
            </div>
          </section>
          <footer className="page-footer"><span>MemoryMate</span><span>Your memories stay yours.</span></footer>
        </div>
      </section>

      <ConversationDialog
        conversation={conversation}
        messages={conversationMessages}
        loading={messagesLoading}
        error={messagesError}
        onClose={() => setConversation(null)}
      />
    </main>
  );
}
