import type {
  CartResponse,
  ChatSessionDetailResponse,
  ChatSessionListResponse,
  DecisionTrace,
  EventReportRequest,
  EventReportResponse,
  HealthStatus,
  ImageUploadResponse,
  LoginRequest,
  LoginResponse,
  ProductCatalogParams,
  ProductCatalogResponse,
  ProductCategoriesResponse,
  ProductDetail,
  RecommendationResponse,
  StreamEvent,
} from "../types";

export const API_BASE_URL = normalizeBaseUrl(import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000");

export function resolveApiUrl(path: string) {
  return new URL(path.startsWith("/") ? path : `/${path}`, API_BASE_URL).toString();
}

export function resolveAssetUrl(url: string) {
  if (!url) return "";
  if (/^(https?:|blob:|data:)/i.test(url)) return url;
  return new URL(url, API_BASE_URL).toString();
}

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(resolveApiUrl(path), {
    headers: {
      Accept: "application/json",
      ...(init?.body ? { "Content-Type": "application/json" } : {}),
      ...(init?.headers ?? {}),
    },
    ...init,
  });
  if (!response.ok) {
    throw new Error(await readError(response));
  }
  return response.json() as Promise<T>;
}

export async function getHealth(): Promise<HealthStatus> {
  return requestJson<HealthStatus>("/health");
}

export async function uploadImage(file: File): Promise<ImageUploadResponse> {
  const form = new FormData();
  form.append("file", file);
  const response = await fetch(resolveApiUrl("/api/images"), {
    method: "POST",
    body: form,
  });
  if (!response.ok) {
    throw new Error(await readError(response));
  }
  return response.json() as Promise<ImageUploadResponse>;
}

export async function login(payload: LoginRequest): Promise<LoginResponse> {
  return requestJson<LoginResponse>("/api/auth/login", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function getProduct(productId: string): Promise<ProductDetail> {
  return requestJson<ProductDetail>(`/api/products/${encodeURIComponent(productId)}`);
}

export async function getProducts(params: ProductCatalogParams = {}): Promise<ProductCatalogResponse> {
  const query = new URLSearchParams();
  if (params.category) query.set("category", params.category);
  if (params.subCategory) query.set("sub_category", params.subCategory);
  if (params.q) query.set("q", params.q);
  if (params.page) query.set("page", String(params.page));
  if (params.pageSize) query.set("page_size", String(params.pageSize));
  if (params.sort) query.set("sort", params.sort);
  return requestJson<ProductCatalogResponse>(`/api/products?${query.toString()}`);
}

export async function getProductCategories(): Promise<ProductCategoriesResponse> {
  return requestJson<ProductCategoriesResponse>("/api/products/categories");
}

export async function getRecommendations(userId: string, size = 24): Promise<RecommendationResponse> {
  const query = new URLSearchParams({ user_id: userId, size: String(size) });
  return requestJson<RecommendationResponse>(`/api/recommendations?${query.toString()}`);
}

export async function getChatSessions(userId: string): Promise<ChatSessionListResponse> {
  const query = new URLSearchParams({ user_id: userId });
  return requestJson<ChatSessionListResponse>(`/api/chat/sessions?${query.toString()}`);
}

export async function getChatSession(userId: string, sessionId: string): Promise<ChatSessionDetailResponse> {
  const query = new URLSearchParams({ user_id: userId });
  return requestJson<ChatSessionDetailResponse>(`/api/chat/sessions/${encodeURIComponent(sessionId)}?${query.toString()}`);
}

export async function getCart(userId: string, sessionId = "all"): Promise<CartResponse> {
  const query = new URLSearchParams({ user_id: userId, session_id: sessionId });
  return requestJson<CartResponse>(`/api/cart?${query.toString()}`);
}

export async function reportEvent(payload: EventReportRequest): Promise<EventReportResponse> {
  return requestJson<EventReportResponse>("/api/events", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function reportChatClick(params: {
  userId: string;
  sessionId: string;
  productId: string;
  query: string;
  source?: string;
}) {
  return reportEvent({
    user_id: params.userId,
    session_id: params.sessionId,
    event_type: "click",
    product_id: params.productId,
    context: {
      from: params.source ?? "chat_card",
      query: params.query.slice(0, 120),
    },
  });
}

export async function reportMallOpen(params: {
  userId: string;
  sessionId: string;
  productId: string;
  position: number;
  brand?: string;
  category?: string;
}) {
  await reportEvent({
    user_id: params.userId,
    session_id: params.sessionId,
    event_type: "click",
    product_id: params.productId,
    position: params.position,
    context: {
      from: "mall",
      page: "home",
      brand: params.brand ?? "",
      category: params.category ?? "",
    },
  });
  return reportEvent({
    user_id: params.userId,
    session_id: params.sessionId,
    event_type: "detail_view",
    product_id: params.productId,
    position: params.position,
    context: {
      from: "mall",
      page: "home",
      brand: params.brand ?? "",
      category: params.category ?? "",
    },
  });
}

export async function reportAddToCart(params: {
  userId: string;
  sessionId: string;
  productId: string;
  source: string;
  sku?: Record<string, string>;
}) {
  return reportEvent({
    user_id: params.userId,
    session_id: params.sessionId,
    event_type: "cart_add",
    product_id: params.productId,
    context: {
      from: params.source,
      ...(params.sku && Object.keys(params.sku).length > 0 ? { sku: params.sku } : {}),
    },
  });
}

export async function reportRemoveFromCart(params: {
  userId: string;
  sessionId: string;
  productId: string;
  source: string;
  sku?: Record<string, string>;
}) {
  return reportEvent({
    user_id: params.userId,
    session_id: params.sessionId,
    event_type: "cart_remove",
    product_id: params.productId,
    context: {
      from: params.source,
      ...(params.sku && Object.keys(params.sku).length > 0 ? { sku: params.sku } : {}),
    },
  });
}

export async function streamChat(
  payload: { user_id: string; session_id: string; message: string; image_id?: string | null },
  onEvent: (event: StreamEvent) => void,
  signal?: AbortSignal,
) {
  const response = await fetch(resolveApiUrl("/api/chat/stream"), {
    method: "POST",
    headers: {
      Accept: "text/event-stream",
      "Content-Type": "application/json",
      "Cache-Control": "no-cache",
    },
    body: JSON.stringify(payload),
    signal,
  });

  if (!response.ok) {
    throw new Error(await readError(response));
  }
  if (!response.body) {
    throw new Error("响应体为空");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  const flushBlock = (block: string) => {
    if (!block.trim()) return;
    let eventName: string | undefined;
    const dataLines: string[] = [];
    for (const line of block.split("\n")) {
      if (line.startsWith("event:")) {
        eventName = line.slice(6).trim();
        continue;
      }
      if (line.startsWith("data:")) {
        dataLines.push(line.slice(5).trimStart());
      }
    }
    if (!eventName && dataLines.length === 0) return;
    const dataText = dataLines.join("\n").trim();
    let payloadObj: Record<string, unknown> = {};
    if (dataText) {
      try {
        payloadObj = JSON.parse(dataText) as Record<string, unknown>;
      } catch {
        payloadObj = { type: eventName ?? "message", raw: dataText };
      }
    }
    const type = (eventName ?? (payloadObj.type as string | undefined) ?? "message") as StreamEvent["type"];
    const event = { ...payloadObj, type } as StreamEvent;
    onEvent(event);
  };

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, "\n");
    let boundary = buffer.indexOf("\n\n");
    while (boundary >= 0) {
      const block = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      flushBlock(block);
      boundary = buffer.indexOf("\n\n");
    }
  }
  buffer += decoder.decode();
  if (buffer.trim()) {
    flushBlock(buffer);
  }
}

function normalizeBaseUrl(url: string) {
  return url.replace(/\/+$/, "");
}

async function readError(response: Response) {
  const text = await response.text().catch(() => "");
  return text ? `HTTP ${response.status}: ${text}` : `HTTP ${response.status}`;
}

export type { DecisionTrace };
