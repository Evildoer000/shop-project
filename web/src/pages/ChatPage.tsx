import { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  Clock,
  ImageUp,
  Loader2,
  MessageSquare,
  Plus,
  RefreshCw,
  Send,
  Sparkles,
  Store,
  X,
} from "lucide-react";
import { MessageBubble } from "../components/MessageBubble";
import { ProductCard } from "../components/ProductCard";
import { TimingPanel } from "../components/TimingPanel";
import { TracePanel } from "../components/TracePanel";
import {
  getChatSession,
  getChatSessions,
  getRecommendations,
  reportAddToCart,
  reportChatClick,
  streamChat,
  uploadImage,
} from "../lib/api";
import { useAppIdentity } from "../lib/app-state";
import { createId } from "../lib/storage";
import type {
  AgentUpdate,
  ChatMessage,
  ChatSessionSummary,
  ChatSessionTurn,
  DecisionTrace,
  ImageUploadResponse,
  ProductCard as ProductCardType,
  RuleEvaluationSummary,
  StreamEvent,
  TimingSpan,
  TimingSummary,
} from "../types";

const quickPrompts = [
  "我是油皮，预算150以内，推荐夏天通勤不闷的防晒",
  "想买一套跑步装备，鞋子和速干衣都要，预算800以内",
  "帮我找一款适合送长辈的低糖零食礼盒",
  "对比一下刚才推荐里前两款，哪款更适合通勤",
];

export function ChatPage() {
  const navigate = useNavigate();
  const { userId, chatSessionId, rotateChatSession, selectChatSession } = useAppIdentity();
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [sessions, setSessions] = useState<ChatSessionSummary[]>([]);
  const [sessionsLoading, setSessionsLoading] = useState(false);
  const [historyError, setHistoryError] = useState("");
  const [loadingSessionId, setLoadingSessionId] = useState("");
  const [input, setInput] = useState("");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [uploadedImage, setUploadedImage] = useState<ImageUploadResponse | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [notice, setNotice] = useState("");
  const [featuredProducts, setFeaturedProducts] = useState<ProductCardType[]>([]);
  const streamControllerRef = useRef<AbortController | null>(null);
  const uploadSeqRef = useRef(0);
  const sessionRef = useRef(chatSessionId);
  const endRef = useRef<HTMLDivElement | null>(null);

  const latestAssistant = useMemo(
    () => [...messages].reverse().find((message) => message.role === "assistant"),
    [messages],
  );
  const lastUserQuery = useMemo(
    () => [...messages].reverse().find((message) => message.role === "user")?.content ?? input,
    [messages, input],
  );

  useEffect(() => {
    void refreshSessions();
  }, [userId]);

  useEffect(() => {
    let ignore = false;
    getRecommendations(userId, 4)
      .then((response) => {
        if (!ignore) setFeaturedProducts(response.products);
      })
      .catch(() => {
        if (!ignore) setFeaturedProducts([]);
      });
    return () => {
      ignore = true;
    };
  }, [userId]);

  useEffect(() => {
    if (sessionRef.current !== chatSessionId) {
      streamControllerRef.current?.abort();
      streamControllerRef.current = null;
      setMessages([]);
      setInput("");
      setNotice("");
      setIsStreaming(false);
      clearImageSelection();
      sessionRef.current = chatSessionId;
    }
  }, [chatSessionId]);

  useEffect(() => {
    if (!selectedFile) {
      setPreviewUrl("");
      return;
    }
    const url = URL.createObjectURL(selectedFile);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [selectedFile]);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages]);

  async function refreshSessions() {
    if (!userId) return;
    setSessionsLoading(true);
    setHistoryError("");
    try {
      const response = await getChatSessions(userId);
      setSessions(response.sessions);
    } catch (error) {
      setHistoryError(error instanceof Error ? error.message : "加载历史会话失败");
    } finally {
      setSessionsLoading(false);
    }
  }

  async function openSession(sessionId: string) {
    if (!sessionId || loadingSessionId) return;
    streamControllerRef.current?.abort();
    streamControllerRef.current = null;
    setIsStreaming(false);
    setNotice("");
    setInput("");
    clearImageSelection();
    sessionRef.current = sessionId;
    selectChatSession(sessionId);
    setLoadingSessionId(sessionId);
    try {
      const response = await getChatSession(userId, sessionId);
      setMessages(messagesFromTurns(response.turns));
    } catch (error) {
      setHistoryError(error instanceof Error ? error.message : "加载会话详情失败");
      setMessages([]);
    } finally {
      setLoadingSessionId("");
    }
  }

  async function handleFileChange(file: File | null) {
    if (!file) return;
    const seq = ++uploadSeqRef.current;
    setSelectedFile(file);
    setUploadedImage(null);
    setUploadError("");
    setUploading(true);
    try {
      const response = await uploadImage(file);
      if (seq === uploadSeqRef.current) {
        setUploadedImage(response);
      }
    } catch (error) {
      if (seq === uploadSeqRef.current) {
        setUploadError(error instanceof Error ? error.message : "上传图片失败");
      }
    } finally {
      if (seq === uploadSeqRef.current) {
        setUploading(false);
      }
    }
  }

  function clearImageSelection() {
    uploadSeqRef.current += 1;
    setSelectedFile(null);
    setPreviewUrl("");
    setUploadedImage(null);
    setUploading(false);
    setUploadError("");
  }

  function startNewChat() {
    streamControllerRef.current?.abort();
    streamControllerRef.current = null;
    const nextSessionId = rotateChatSession();
    sessionRef.current = nextSessionId;
    setMessages([]);
    setInput("");
    setNotice("");
    setIsStreaming(false);
    clearImageSelection();
  }

  async function sendMessage() {
    const text = input.trim();
    if (isStreaming || uploading) return;
    if (!text && !uploadedImage) return;
    if (selectedFile && !uploadedImage) {
      setUploadError("图片还没有上传成功，请稍后再发送。");
      return;
    }

    const userMessage: ChatMessage = {
      id: createId("user"),
      role: "user",
      content: text || "请根据这张图片推荐相似商品",
      attachedImageUrl: uploadedImage?.image_url ?? null,
    };
    const assistantId = createId("assistant");
    const assistantMessage: ChatMessage = {
      id: assistantId,
      role: "assistant",
      content: "",
      isStreaming: true,
      traceLogs: [],
      agentUpdates: [],
      timingSpans: [],
      timingSummary: null,
      ruleEvaluation: null,
      products: [],
      decisionTrace: null,
    };

    const payload = {
      user_id: userId,
      session_id: chatSessionId,
      message: text,
      image_id: uploadedImage?.image_id ?? null,
    };

    setMessages((current) => [...current, userMessage, assistantMessage]);
    setInput("");
    clearImageSelection();
    setIsStreaming(true);
    setNotice("");

    streamControllerRef.current?.abort();
    const controller = new AbortController();
    streamControllerRef.current = controller;

    try {
      await streamChat(payload, (event) => handleStreamEvent(assistantId, event), controller.signal);
    } catch (error) {
      if (!controller.signal.aborted) {
        updateAssistant(assistantId, (message) => ({
          ...message,
          isStreaming: false,
          isError: true,
          content: message.content || `出错了：${error instanceof Error ? error.message : "流式连接失败"}`,
        }));
      }
    } finally {
      if (streamControllerRef.current === controller) {
        streamControllerRef.current = null;
      }
      setIsStreaming(false);
      updateAssistant(assistantId, (message) => ({ ...message, isStreaming: false }));
      window.setTimeout(() => {
        void refreshSessions();
      }, 600);
    }
  }

  function handleStreamEvent(assistantId: string, event: StreamEvent) {
    switch (event.type) {
      case "trace":
        updateAssistant(assistantId, (message) => ({
          ...message,
          traceLogs: [
            ...(message.traceLogs ?? []),
            { stage: String(event.stage ?? ""), content: String(event.content ?? "") },
          ],
        }));
        break;
      case "decision_trace":
        updateAssistant(assistantId, (message) => ({
          ...message,
          decisionTrace: isDecisionTrace(event.trace) ? event.trace : null,
        }));
        break;
      case "agent_update":
        updateAssistant(assistantId, (message) => ({
          ...message,
          agentUpdates: mergeAgentUpdate(message.agentUpdates ?? [], {
            stage: String(event.stage ?? "agent"),
            title: String(event.title ?? event.stage ?? "agent"),
            contentDelta: String(event.content_delta ?? ""),
            done: Boolean(event.done),
          }),
        }));
        break;
      case "timing_update":
        const timingSpan = isTimingSpan(event.span) ? event.span : null;
        const timingSummary = isTimingSummary(event.summary) ? event.summary : null;
        const ruleEvaluation = isRuleEvaluation(event.evaluation) ? event.evaluation : null;
        updateAssistant(assistantId, (message) => ({
          ...message,
          timingSpans: timingSpan
            ? [...(message.timingSpans ?? []), timingSpan]
            : message.timingSpans ?? [],
          timingSummary: timingSummary ?? message.timingSummary ?? null,
          ruleEvaluation: ruleEvaluation ?? message.ruleEvaluation ?? null,
        }));
        break;
      case "token":
        updateAssistant(assistantId, (message) => ({
          ...message,
          content: `${message.content}${event.content ?? ""}`,
        }));
        break;
      case "product_cards":
        updateAssistant(assistantId, (message) => ({ ...message, products: asProductList(event.products) }));
        break;
      case "error":
        updateAssistant(assistantId, (message) => ({
          ...message,
          isError: true,
          isStreaming: false,
          content: message.content || `出错了：${event.message ?? event.error ?? "服务端返回错误"}`,
        }));
        break;
      case "done":
        updateAssistant(assistantId, (message) => ({ ...message, isStreaming: false }));
        break;
      default:
        break;
    }
  }

  function updateAssistant(id: string, updater: (message: ChatMessage) => ChatMessage) {
    setMessages((current) => current.map((message) => (message.id === id ? updater(message) : message)));
  }

  async function openProduct(productId: string) {
    reportChatClick({
      userId,
      sessionId: chatSessionId,
      productId,
      query: lastUserQuery,
    }).catch(() => undefined);
    navigate(`/products/${productId}`);
  }

  async function addProductToCart(productId: string) {
    try {
      await reportAddToCart({
        userId,
        sessionId: chatSessionId,
        productId,
        source: "chat_card",
      });
      setNotice("已加入购物车");
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "加入购物车失败");
    }
  }

  async function addFeaturedToCart(productId: string) {
    try {
      await reportAddToCart({
        userId,
        sessionId: chatSessionId,
        productId,
        source: "chat_side",
      });
      setNotice("已加入购物车");
    } catch {
      setNotice("加入购物车失败");
    }
  }

  return (
    <div className="page-grid chat-grid">
      <section className="panel chat-panel">
        <div className="page-heading page-heading-row chat-heading">
          <div>
            <p className="eyebrow">AI Shopping Assistant</p>
            <h1>导购聊天</h1>
            <p>支持文本、图片和多轮追问。当前会话会在发送消息后自动保存到历史记录。</p>
          </div>
          <button className="button button-secondary" type="button" onClick={startNewChat}>
            <Plus size={16} />
            添加新会话
          </button>
        </div>

        <div className="message-list">
          {loadingSessionId ? (
            <div className="chat-empty">
              <Loader2 className="spin" size={28} />
              <h2>正在加载历史会话...</h2>
            </div>
          ) : messages.length === 0 ? (
            <div className="chat-empty">
              <Sparkles size={28} />
              <h2>描述你的需求，导购 Agent 会拆解约束并检索商品。</h2>
              <div className="prompt-grid">
                {quickPrompts.map((prompt) => (
                  <button key={prompt} className="prompt-chip" type="button" onClick={() => setInput(prompt)}>
                    {prompt}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            messages.map((message) => (
              <MessageBubble
                key={message.id}
                message={message}
                onProductOpen={openProduct}
                onProductAddToCart={addProductToCart}
              />
            ))
          )}
          <div ref={endRef} />
        </div>

        {notice ? <div className="notice-line">{notice}</div> : null}

        <div className="composer">
          {selectedFile ? (
            <div className="upload-preview">
              {previewUrl ? <img src={previewUrl} alt="待发送图片" /> : null}
              <div>
                <strong>{selectedFile.name}</strong>
                <span>{uploading ? "正在上传..." : uploadedImage ? "上传完成" : uploadError || "等待上传"}</span>
              </div>
              {uploading ? <Loader2 className="spin" size={18} /> : null}
              <button className="icon-button" type="button" onClick={clearImageSelection} title="移除图片">
                <X size={16} />
              </button>
            </div>
          ) : null}

          <div className="composer-row">
            <label className="icon-button file-button" title="上传图片">
              <ImageUp size={18} />
              <input
                type="file"
                accept="image/jpeg,image/png,image/webp"
                onChange={(event) => handleFileChange(event.target.files?.[0] ?? null)}
              />
            </label>
            <textarea
              value={input}
              onChange={(event) => setInput(event.target.value)}
              placeholder="例如：预算150以内，适合油皮夏天通勤的防晒，不要酒精味重"
              rows={2}
              onKeyDown={(event) => {
                if (event.key === "Enter" && !event.shiftKey) {
                  event.preventDefault();
                  sendMessage();
                }
              }}
            />
            <button
              className="button button-primary send-button"
              type="button"
              onClick={sendMessage}
              disabled={isStreaming || uploading || loadingSessionId !== "" || (!input.trim() && !uploadedImage)}
            >
              {isStreaming ? <Loader2 className="spin" size={18} /> : <Send size={18} />}
              发送
            </button>
          </div>
        </div>
      </section>

      <aside className="side-stack">
        <section className="panel session-panel">
          <div className="panel-title session-panel-title">
            <span>
              <MessageSquare size={16} />
              历史会话
            </span>
            <button className="icon-button small-icon-button" type="button" onClick={() => void refreshSessions()} title="刷新历史">
              {sessionsLoading ? <Loader2 className="spin" size={14} /> : <RefreshCw size={14} />}
            </button>
          </div>
          {historyError ? <div className="error-line compact-line">{historyError}</div> : null}
          <div className="session-list">
            {sessions.length === 0 && !sessionsLoading ? (
              <div className="empty-hint">还没有历史会话。发送第一条消息后，这里会自动出现记录。</div>
            ) : null}
            {sessions.map((session) => (
              <button
                key={session.session_id}
                className={`session-item ${session.session_id === chatSessionId ? "active" : ""}`}
                type="button"
                onClick={() => openSession(session.session_id)}
                disabled={loadingSessionId === session.session_id}
              >
                <strong>{session.title || "未命名会话"}</strong>
                <span>{session.last_message || "暂无回复内容"}</span>
                <small>
                  <Clock size={12} />
                  {session.turn_count} 轮 · {formatSessionTime(session.updated_at)}
                </small>
              </button>
            ))}
          </div>
        </section>

        <TimingPanel
          spans={latestAssistant?.timingSpans}
          summary={latestAssistant?.timingSummary}
          evaluation={latestAssistant?.ruleEvaluation}
        />

        <TracePanel trace={latestAssistant?.decisionTrace} />

        <section className="panel">
          <div className="panel-title">
            <Store size={16} />
            <span>热门商品预览</span>
          </div>
          <div className="mini-product-list">
            {featuredProducts.length === 0 ? (
              <div className="empty-hint">暂无推荐预览。</div>
            ) : (
              featuredProducts.map((product) => (
                <ProductCard
                  key={product.product_id}
                  product={product}
                  compact
                  showScore
                  onOpen={() => navigate(`/products/${product.product_id}`)}
                  onAdd={() => addFeaturedToCart(product.product_id)}
                />
              ))
            )}
          </div>
        </section>
      </aside>
    </div>
  );
}

function messagesFromTurns(turns: ChatSessionTurn[]): ChatMessage[] {
  return turns.flatMap((turn) => [
    {
      id: `turn-${turn.turn_id}-user`,
      role: "user" as const,
      content: turn.user_message,
    },
    {
      id: `turn-${turn.turn_id}-assistant`,
      role: "assistant" as const,
      content: turn.assistant_message,
      products: turn.products ?? [],
      decisionTrace: traceFromTurn(turn),
      timingSpans: [],
      timingSummary: null,
      ruleEvaluation: null,
      traceLogs: [],
      agentUpdates: [],
    },
  ]);
}

function traceFromTurn(turn: ChatSessionTurn): DecisionTrace | null {
  const hasTrace = Object.keys(turn.trace_summary ?? {}).length > 0 || Object.keys(turn.rewrite_summary ?? {}).length > 0 || turn.route;
  if (!hasTrace) return null;
  return {
    route: turn.route,
    planner_proposal: turn.rewrite_summary,
    retrieval_summary: turn.trace_summary,
  };
}

function mergeAgentUpdate(current: AgentUpdate[], next: AgentUpdate): AgentUpdate[] {
  const index = current.findIndex((item) => item.stage === next.stage);
  if (index < 0) return [...current, next];
  return current.map((item, itemIndex) => (
    itemIndex === index
      ? {
          ...item,
          title: next.title || item.title,
          contentDelta: `${item.contentDelta}${next.contentDelta}`,
          done: item.done || next.done,
        }
      : item
  ));
}

function isDecisionTrace(value: unknown): value is DecisionTrace {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function isTimingSpan(value: unknown): value is TimingSpan {
  return value !== null && typeof value === "object" && !Array.isArray(value) && "name" in value && "duration_ms" in value;
}

function isTimingSummary(value: unknown): value is TimingSummary {
  return value !== null && typeof value === "object" && !Array.isArray(value) && "completed_spans" in value;
}

function isRuleEvaluation(value: unknown): value is RuleEvaluationSummary {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function asProductList(value: unknown): ProductCardType[] {
  return Array.isArray(value) ? (value as ProductCardType[]) : [];
}

function formatSessionTime(value?: string | null) {
  if (!value) return "未知时间";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "未知时间";
  const now = Date.now();
  const diffMinutes = Math.floor((now - date.getTime()) / 60000);
  if (diffMinutes < 1) return "刚刚";
  if (diffMinutes < 60) return `${diffMinutes} 分钟前`;
  const diffHours = Math.floor(diffMinutes / 60);
  if (diffHours < 24) return `${diffHours} 小时前`;
  return date.toLocaleDateString("zh-CN", { month: "2-digit", day: "2-digit" });
}
