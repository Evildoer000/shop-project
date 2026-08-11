import { useEffect, useState } from "react";
import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import { AppShell } from "./components/AppShell";
import { AppIdentityProvider, useAppIdentity } from "./lib/app-state";
import { getHealth } from "./lib/api";
import { CartPage } from "./pages/CartPage";
import { ChatPage } from "./pages/ChatPage";
import { LoginPage } from "./pages/LoginPage";
import { MallPage } from "./pages/MallPage";
import { ObservabilityPage } from "./pages/ObservabilityPage";
import { ProductDetailPage } from "./pages/ProductDetailPage";
import type { HealthStatus } from "./types";

export default function App() {
  return (
    <AppIdentityProvider>
      <BrowserRouter>
        <AppRoutes />
      </BrowserRouter>
    </AppIdentityProvider>
  );
}

function AppRoutes() {
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const { isAuthenticated } = useAppIdentity();

  useEffect(() => {
    let cancelled = false;

    async function probe() {
      try {
        const response = await getHealth();
        if (!cancelled) setHealth(response);
      } catch {
        if (!cancelled) setHealth(null);
      }
    }

    probe();
    const timer = window.setInterval(probe, 30000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, []);

  return isAuthenticated ? (
    <AppShell health={health}>
      <Routes>
        <Route path="/" element={<ChatPage />} />
        <Route path="/mall" element={<MallPage />} />
        <Route path="/cart" element={<CartPage />} />
        <Route path="/observability" element={<ObservabilityPage />} />
        <Route path="/products/:productId" element={<ProductDetailPage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </AppShell>
  ) : (
    <LoginPage health={health} />
  );
}
