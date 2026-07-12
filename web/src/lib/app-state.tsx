import { createContext, useContext, useMemo } from "react";
import { login as requestLogin } from "./api";
import { createId, usePersistentState } from "./storage";
import type { LoginResponse } from "../types";

type AppIdentity = {
  authUser: LoginResponse | null;
  isAuthenticated: boolean;
  userId: string;
  login: (phone: string, password: string) => Promise<void>;
  logout: () => void;
  chatSessionId: string;
  rotateChatSession: () => string;
  selectChatSession: (sessionId: string) => void;
  mallSessionId: string;
  cartSessionId: string;
  productSessionId: string;
};

const AppIdentityContext = createContext<AppIdentity | null>(null);

export function AppIdentityProvider({ children }: { children: React.ReactNode }) {
  const [authUser, setAuthUser] = usePersistentState<LoginResponse | null>("shop-agent:auth-user", null);
  const [chatSessionId, setChatSessionId] = usePersistentState("shop-agent:chat-session-id", createId("chat"));

  const value = useMemo<AppIdentity>(() => ({
    authUser,
    isAuthenticated: authUser != null,
    userId: authUser?.user_id ?? "",
    login: async (phone: string, password: string) => {
      const user = await requestLogin({ phone, password });
      setAuthUser(user);
      setChatSessionId(createId("chat"));
    },
    logout: () => {
      setAuthUser(null);
      setChatSessionId(createId("chat"));
    },
    chatSessionId,
    rotateChatSession: () => {
      const sessionId = createId("chat");
      setChatSessionId(sessionId);
      return sessionId;
    },
    selectChatSession: (sessionId: string) => setChatSessionId(sessionId),
    mallSessionId: "mall_session",
    cartSessionId: "cart",
    productSessionId: "product_detail",
  }), [authUser, setAuthUser, chatSessionId, setChatSessionId]);

  return <AppIdentityContext.Provider value={value}>{children}</AppIdentityContext.Provider>;
}

export function useAppIdentity() {
  const value = useContext(AppIdentityContext);
  if (!value) {
    throw new Error("useAppIdentity must be used inside AppIdentityProvider");
  }
  return value;
}
