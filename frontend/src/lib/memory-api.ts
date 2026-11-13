export type MemoryType = "episodic" | "semantic" | "consolidated";

export type Memory = {
  id: string;
  memory_type: MemoryType;
  content: string;
  confidence: number;
  tags: string[];
  source_conversation_id: string | null;
  is_active: boolean;
  access_count: number;
  created_at: string;
  updated_at: string;
};

export type SearchHit = {
  id: string;
  content: string;
  type: MemoryType;
  confidence: number;
  similarity: number;
  created_at: string;
  tags: string[];
};

export type GraphNode = {
  id: string;
  label: string;
  type: string;
};

export type GraphEdge = {
  source: string;
  target: string;
  label: string;
};

export type KnowledgeGraph = {
  nodes: GraphNode[];
  edges: GraphEdge[];
  available: boolean;
};

export type Conversation = {
  id: string;
  title: string;
  summary: string | null;
  created_at: string;
  updated_at: string;
  message_count: number;
};

export type ConversationMessage = {
  id: string;
  role: string;
  content: string;
  created_at: string;
  token_count: number | null;
};

export type MemoryStats = {
  total_memories: number;
  episodic_count: number;
  semantic_count: number;
  consolidated_count: number;
  avg_confidence: number;
};

export type ChatContextRecord = Record<string, unknown>;

export type ChatRequest = {
  user_id: string;
  conversation_id: string | null;
  message: string;
  use_memory: boolean;
  use_graph: boolean;
};

export type ChatResponse = {
  conversation_id: string;
  message_id: string;
  response: string;
  retrieved_memories: ChatContextRecord[];
  retrieved_graph_context: ChatContextRecord[];
  tokens_used: number | null;
  latency_ms: number | null;
};

const API_URL = "/api";

export const USER_ID =
  process.env.NEXT_PUBLIC_MEMORYMATE_USER_ID ?? "default-user";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_URL}${path}`, {
      ...init,
      headers: {
        ...(init?.body ? { "Content-Type": "application/json" } : {}),
        ...init?.headers,
      },
    });
  } catch {
    throw new Error("Cannot reach the MemoryMate API. Check that the backend is running.");
  }

  if (!response.ok) {
    let detail = `Request failed (${response.status})`;
    try {
      const body = (await response.json()) as { detail?: string };
      if (body.detail) detail = body.detail;
    } catch {
      // Keep the HTTP status message when the server does not return JSON.
    }
    throw new Error(detail);
  }
  return (await response.json()) as T;
}

export function getMemories(): Promise<Memory[]> {
  return request(`/memory/${encodeURIComponent(USER_ID)}?limit=200`);
}

export function getMemoryStats(): Promise<MemoryStats> {
  return request(`/memory/${encodeURIComponent(USER_ID)}/stats`);
}

export function searchMemories(query: string): Promise<SearchHit[]> {
  return request("/memory/search", {
    method: "POST",
    body: JSON.stringify({
      user_id: USER_ID,
      query,
      top_k: 100,
      memory_types: ["episodic", "semantic", "consolidated"],
      min_confidence: 0,
    }),
  });
}

export function getKnowledgeGraph(): Promise<KnowledgeGraph> {
  return request(`/graph/${encodeURIComponent(USER_ID)}`);
}

export function getConversations(): Promise<Conversation[]> {
  return request(`/chat/conversations/${encodeURIComponent(USER_ID)}?limit=100`);
}

export function getConversationMessages(
  conversationId: string,
): Promise<ConversationMessage[]> {
  return request(
    `/chat/conversations/${encodeURIComponent(USER_ID)}/${encodeURIComponent(conversationId)}/messages`,
  );
}

export function postChatMessage(payload: ChatRequest): Promise<ChatResponse> {
  return request("/chat", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function updateMemory(
  memoryId: string,
  content: string,
): Promise<Memory> {
  return request(
    `/memory/${encodeURIComponent(USER_ID)}/${encodeURIComponent(memoryId)}`,
    {
      method: "PATCH",
      body: JSON.stringify({ content }),
    },
  );
}

export function deleteMemory(memoryId: string): Promise<{ deleted: boolean }> {
  return request(
    `/memory/${encodeURIComponent(USER_ID)}/${encodeURIComponent(memoryId)}`,
    { method: "DELETE" },
  );
}
