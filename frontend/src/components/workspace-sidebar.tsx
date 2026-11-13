import Link from "next/link";

type WorkspaceSidebarProps = {
  active: "explorer" | "chat";
  connected?: boolean;
};

function SidebarIcon({ kind }: { kind: "spark" | "network" | "chat" }) {
  if (kind === "chat") {
    return (
      <svg aria-hidden="true" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round">
        <path d="M20 11.5a7.5 7.5 0 0 1-7.5 7.5H6l-3 2 1-4.1A7.5 7.5 0 1 1 20 11.5Z" />
        <path d="M8 11h8M8 14h5" />
      </svg>
    );
  }
  if (kind === "network") {
    return (
      <svg aria-hidden="true" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round">
        <circle cx="12" cy="5" r="2" /><circle cx="5" cy="18" r="2" /><circle cx="19" cy="18" r="2" />
        <path d="m11 7-5 9m7-9 5 9M7 18h10" />
      </svg>
    );
  }
  return (
    <svg aria-hidden="true" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round">
      <path d="m12 3 1.9 5.8L20 11l-6.1 2.2L12 19l-1.9-5.8L4 11l6.1-2.2L12 3Z" />
      <path d="m19 14 1 2.5 2.5 1-2.5 1L19 21l-1-2.5-2.5-1 2.5-1L19 14Z" />
    </svg>
  );
}

export default function WorkspaceSidebar({ active, connected }: WorkspaceSidebarProps) {
  return (
    <aside className="sidebar">
      <Link href="/" className="brand" aria-label="MemoryMate home">
        <span className="brand-mark"><SidebarIcon kind="spark" /></span>
        <span>memory<span className="brand-light">mate</span></span>
      </Link>
      <div className="workspace-label">WORKSPACE</div>
      <nav className="side-nav" aria-label="Main navigation">
        <Link className={`nav-item${active === "explorer" ? " active" : ""}`} href="/#memories">
          <SidebarIcon kind="spark" /> Memory explorer
        </Link>
        <Link className="nav-item" href="/#knowledge-graph">
          <SidebarIcon kind="network" /> Knowledge graph
        </Link>
        <Link className={`nav-item${active === "chat" ? " active" : ""}`} href="/chat">
          <SidebarIcon kind="chat" /> Chat
        </Link>
      </nav>
      <div className="sidebar-bottom">
        {connected !== undefined ? (
          <div className={`connection-indicator${connected ? "" : " disconnected"}`}>
            <span />{connected ? "Memory API connected" : "Memory API disconnected"}
          </div>
        ) : null}
        <div className="sidebar-caption">Your personal knowledge, thoughtfully organized.</div>
      </div>
    </aside>
  );
}
