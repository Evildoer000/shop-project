import { Bot, CheckCircle2, Clock3, FileText, UserRound } from "lucide-react";
import { ProductCard } from "./ProductCard";
import type { ChatMessage } from "../types";
import { resolveAssetUrl } from "../lib/api";

type MessageBubbleProps = {
  message: ChatMessage;
  onProductOpen: (productId: string) => void;
  onProductAddToCart: (productId: string) => void;
};

export function MessageBubble({ message, onProductOpen, onProductAddToCart }: MessageBubbleProps) {
  const isUser = message.role === "user";
  const products = message.products ?? [];
  const updates = message.agentUpdates ?? [];
  const traceLogs = message.traceLogs ?? [];

  return (
    <div className={`message-item ${isUser ? "message-user" : "message-assistant"}`}>
      <div className="message-avatar">
        {isUser ? <UserRound size={18} /> : <Bot size={18} />}
      </div>

      <div className="message-main">
        <div className={`message-bubble ${message.isError ? "message-error" : ""}`}>
          {message.attachedImageUrl ? (
            <img
              className="message-image"
              src={resolveAssetUrl(message.attachedImageUrl)}
              alt="用户上传图片"
            />
          ) : null}

          {updates.length > 0 && !isUser ? (
            <div className="agent-update-list">
              {updates.map((update) => (
                <div key={update.stage} className="agent-update">
                  <span className={`agent-update-icon ${update.done ? "done" : ""}`}>
                    {update.done ? <CheckCircle2 size={14} /> : <Clock3 size={14} />}
                  </span>
                  <div>
                    <div className="agent-update-title">{update.title || update.stage}</div>
                    <div className="agent-update-content">{update.contentDelta}</div>
                  </div>
                </div>
              ))}
            </div>
          ) : null}

          {message.content ? (
            <p className="message-text">{message.content}</p>
          ) : message.isStreaming ? (
            <div className="typing-line">
              <span />
              <span />
              <span />
            </div>
          ) : null}
        </div>

        {products.length > 0 ? (
          <div className="message-products">
            {products.map((product) => (
              <ProductCard
                key={product.product_id}
                product={product}
                compact
                onOpen={() => onProductOpen(product.product_id)}
                onAdd={() => onProductAddToCart(product.product_id)}
              />
            ))}
          </div>
        ) : null}

        {!isUser && (traceLogs.length > 0 || message.decisionTrace) ? (
          <details className="message-trace">
            <summary>
              <FileText size={14} />
              <span>查看轨迹</span>
            </summary>
            {traceLogs.length > 0 ? (
              <div className="trace-log-list">
                {traceLogs.map((trace, index) => (
                  <div key={`${trace.stage}-${index}`} className="trace-log-row">
                    <span>{trace.stage}</span>
                    <p>{trace.content}</p>
                  </div>
                ))}
              </div>
            ) : null}
            {message.decisionTrace ? (
              <pre className="inline-json">{JSON.stringify(message.decisionTrace, null, 2)}</pre>
            ) : null}
          </details>
        ) : null}
      </div>
    </div>
  );
}
