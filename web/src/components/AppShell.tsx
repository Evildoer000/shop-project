import { NavLink } from "react-router-dom";
import { Activity, Bot, HeartHandshake, LogOut, Phone, ShoppingCart, Store, Wifi, WifiOff, RefreshCw } from "lucide-react";
import { useAppIdentity } from "../lib/app-state";
import type { HealthStatus } from "../types";

type AppShellProps = {
  health: HealthStatus | null;
  children: React.ReactNode;
};

const navItems = [
  { to: "/", label: "聊天导购", icon: Bot },
  { to: "/mall", label: "商城推荐", icon: Store },
  { to: "/cart", label: "购物车", icon: ShoppingCart },
  { to: "/observability", label: "运行观测", icon: Activity },
];

export function AppShell({ health, children }: AppShellProps) {
  const { authUser, userId, chatSessionId, rotateChatSession, logout } = useAppIdentity();
  const healthy = health?.status === "ok";

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">S</div>
          <div>
            <div className="brand-title">Shop Agent</div>
            <div className="brand-subtitle">Web 导购助手</div>
          </div>
        </div>

        <nav className="nav-list">
          {navItems.map(({ to, label, icon: Icon }) => (
            <NavLink key={to} to={to} end={to === "/"} className={({ isActive }) => `nav-link ${isActive ? "active" : ""}`}>
              <Icon size={18} />
              <span>{label}</span>
            </NavLink>
          ))}
        </nav>

        <div className="sidebar-card">
          <div className="sidebar-card-title">
            <HeartHandshake size={16} />
            <span>当前用户</span>
          </div>
          <div className="login-user-card">
            <Phone size={15} />
            <div>
              <strong>{authUser?.display_name ?? "未登录"}</strong>
              <span>{authUser?.phone ?? "-"}</span>
            </div>
          </div>
          <div className="chip-row">
            <span className="chip chip-dark">user: {userId}</span>
            <span className="chip chip-dark">chat: {chatSessionId}</span>
          </div>
          <button className="button button-dark button-block" onClick={rotateChatSession}>
            <RefreshCw size={16} />
            新会话
          </button>
          <button className="button button-dark button-block" onClick={logout}>
            <LogOut size={16} />
            退出登录
          </button>
        </div>
      </aside>

      <main className="main-shell">
        <header className="topbar">
          <div className="topbar-left">
            <div className={`status-pill ${healthy ? "status-ok" : "status-bad"}`}>
              {healthy ? <Wifi size={14} /> : <WifiOff size={14} />}
              <span>{healthy ? "后端在线" : "后端未连通"}</span>
            </div>
            <span className="topbar-note">API: {import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000"}</span>
          </div>
          <div className="topbar-right">
            <span className="topbar-note">登录用户</span>
            <strong>{authUser?.phone ?? userId}</strong>
          </div>
        </header>

        <div className="page-shell">
          {children}
        </div>
      </main>
    </div>
  );
}
