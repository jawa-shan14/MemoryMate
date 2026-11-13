import type { Metadata } from "next";
import ChatClient from "./chat-client";

export const metadata: Metadata = {
  title: "Chat | MemoryMate",
  description: "Talk with MemoryMate and see the context it actually retrieves.",
};

export default function ChatPage() {
  return <ChatClient />;
}
