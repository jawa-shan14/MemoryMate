import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import MemoryExplorer from "./memory-explorer";

const episodicMemory = {
  id: "memory-episodic",
  memory_type: "episodic",
  content: "Discussed a launch plan with the design team.",
  confidence: 0.96,
  tags: ["work", "launch"],
  source_conversation_id: "conversation-123",
  is_active: true,
  access_count: 1,
  created_at: "2026-02-20T10:00:00Z",
  updated_at: "2026-02-20T10:00:00Z",
};

const semanticMemory = {
  ...episodicMemory,
  id: "memory-semantic",
  memory_type: "semantic",
  content: "Prefers concise product updates.",
  tags: ["preference"],
  source_conversation_id: null,
};

const graph = {
  available: true,
  nodes: [
    { id: "entity-1", label: "Design team", type: "Organization" },
    { id: "entity-2", label: "Product launch", type: "Project" },
  ],
  edges: [{ source: "entity-1", target: "entity-2", label: "WORKING_ON" }],
};

function installApiMock() {
  const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    if (url.includes("/memory/search")) {
      return Response.json([{ id: "memory-episodic", content: episodicMemory.content }]);
    }
    if (url.includes("/memory/default-user/stats")) {
      return Response.json({ total_memories: 2, episodic_count: 1, semantic_count: 1, consolidated_count: 0, avg_confidence: 0.96 });
    }
    if (url.includes("/memory/default-user/memory-episodic") && init?.method === "PATCH") {
      return Response.json({ ...episodicMemory, content: "Updated launch plan." });
    }
    if (url.includes("/memory/default-user/memory-semantic") && init?.method === "DELETE") {
      return Response.json({ deleted: true });
    }
    if (url.includes("/memory/default-user")) return Response.json([episodicMemory, semanticMemory]);
    if (url.includes("/graph/default-user")) return Response.json(graph);
    if (url.includes("/chat/conversations/default-user/conversation-123/messages")) {
      return Response.json([{
        id: "message-1",
        role: "user",
        content: "Let's map out the launch.",
        created_at: "2026-02-20T09:00:00Z",
        token_count: 7,
      }]);
    }
    if (url.includes("/chat/conversations/default-user")) {
      return Response.json([{
        id: "conversation-123",
        title: "Launch planning",
        summary: null,
        created_at: "2026-02-20T09:00:00Z",
        updated_at: "2026-02-20T10:00:00Z",
        message_count: 2,
      }]);
    }
    throw new Error(`Unexpected API request: ${url}`);
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

describe("MemoryExplorer", () => {
  beforeEach(() => {
    installApiMock();
  });

  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it("loads episodic and semantic memories separately and renders graph entities", async () => {
    render(<MemoryExplorer />);

    expect(await screen.findByText("Discussed a launch plan with the design team.")).toBeTruthy();
    expect(screen.getByText("Prefers concise product updates.")).toBeTruthy();
    expect(screen.getByRole("heading", { name: "Episodic" })).toBeTruthy();
    expect(screen.getByRole("heading", { name: "Semantic" })).toBeTruthy();
    expect(await screen.findByRole("button", { name: "Select Design team, Organization" })).toBeTruthy();
    expect(screen.getAllByText("WORKING ON").length).toBeGreaterThan(0);
  });

  it("uses backend semantic search and filters the displayed memories", async () => {
    const user = userEvent.setup();
    const fetchMock = vi.mocked(fetch);
    render(<MemoryExplorer />);
    await screen.findByText("Prefers concise product updates.");

    await user.type(screen.getByRole("textbox", { name: "Search memories" }), "launch");

    await waitFor(() => {
      expect(screen.queryByText("Prefers concise product updates.")).toBeNull();
    });
    expect(await screen.findByText("Discussed a launch plan with the design team.")).toBeTruthy();
    const searchRequest = fetchMock.mock.calls.find(([input]) => String(input).includes("/memory/search"));
    expect(searchRequest).toBeTruthy();
    expect(JSON.parse(String(searchRequest?.[1]?.body))).toMatchObject({ query: "launch", user_id: "default-user" });
  });

  it("persists an edited memory through the API", async () => {
    const user = userEvent.setup();
    const fetchMock = vi.mocked(fetch);
    render(<MemoryExplorer />);
    await screen.findByText(episodicMemory.content);

    await user.click(screen.getByRole("button", { name: `Edit memory: ${episodicMemory.content}` }));
    const editor = screen.getByRole("textbox", { name: "Edit memory content" });
    await user.clear(editor);
    await user.type(editor, "Updated launch plan.");
    await user.click(screen.getByRole("button", { name: "Save changes" }));

    expect(await screen.findByText("Updated launch plan.")).toBeTruthy();
    const patchRequest = fetchMock.mock.calls.find(([, init]) => init?.method === "PATCH");
    expect(patchRequest).toBeTruthy();
    expect(JSON.parse(String(patchRequest?.[1]?.body))).toEqual({ content: "Updated launch plan." });
  });

  it("persists deletion and removes the memory from the library", async () => {
    const user = userEvent.setup();
    const fetchMock = vi.mocked(fetch);
    vi.stubGlobal("confirm", vi.fn(() => true));
    render(<MemoryExplorer />);
    await screen.findByText(semanticMemory.content);

    await user.click(screen.getByRole("button", { name: `Delete memory: ${semanticMemory.content}` }));

    await waitFor(() => expect(screen.queryByText(semanticMemory.content)).toBeNull());
    expect(fetchMock.mock.calls.some(([, init]) => init?.method === "DELETE")).toBe(true);
  });

  it("opens the memory's source conversation from the backend", async () => {
    const user = userEvent.setup();
    const fetchMock = vi.mocked(fetch);
    render(<MemoryExplorer />);
    await screen.findByText(episodicMemory.content);

    await user.click(screen.getByRole("button", { name: /Launch planning/ }));

    expect(await screen.findByRole("dialog")).toBeTruthy();
    expect(screen.getByText("Let's map out the launch.")).toBeTruthy();
    expect(fetchMock.mock.calls.some(([input]) => String(input).includes("/conversation-123/messages"))).toBe(true);
  });

  it("shows a clear empty state when the backend has no memories", async () => {
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.includes("/memory/default-user/stats")) {
        return Response.json({ total_memories: 0, episodic_count: 0, semantic_count: 0, consolidated_count: 0, avg_confidence: 0 });
      }
      if (url.includes("/memory/default-user")) return Response.json([]);
      if (url.includes("/graph/default-user")) return Response.json({ available: true, nodes: [], edges: [] });
      if (url.includes("/chat/conversations/default-user")) return Response.json([]);
      throw new Error(`Unexpected API request: ${url}`);
    }));
    render(<MemoryExplorer />);

    expect(await screen.findByText("Your memory is taking shape")).toBeTruthy();
    expect(screen.getByText("No entities yet")).toBeTruthy();
  });

  it("shows API failures instead of presenting a successful empty state", async () => {
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.includes("/memory/default-user?")) {
        return Response.json({ detail: "Database unavailable" }, { status: 503 });
      }
      if (url.includes("/memory/default-user/stats")) {
        return Response.json({ detail: "Database unavailable" }, { status: 503 });
      }
      if (url.includes("/graph/default-user")) return Response.json({ available: false, nodes: [], edges: [] });
      if (url.includes("/chat/conversations/default-user")) return Response.json([]);
      throw new Error(`Unexpected API request: ${url}`);
    }));
    render(<MemoryExplorer />);

    expect((await screen.findAllByRole("alert")).length).toBeGreaterThan(0);
    expect(screen.getAllByText("Database unavailable").length).toBeGreaterThan(0);
    expect(screen.getByText("Memory API disconnected")).toBeTruthy();
    expect(screen.queryByText("Your memory is taking shape")).toBeNull();
  });
});
