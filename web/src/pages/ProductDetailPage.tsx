import { useEffect, useMemo, useState } from "react";
import type { ReactNode } from "react";
import { useNavigate, useParams } from "react-router-dom";
import {
  ArrowLeft,
  BadgeCheck,
  Info,
  Loader2,
  MessageSquareText,
  PackageCheck,
  ShoppingCart,
  Star,
  Tags,
} from "lucide-react";
import { getProduct, reportAddToCart, resolveAssetUrl } from "../lib/api";
import { useAppIdentity } from "../lib/app-state";
import type { ProductDetail } from "../types";

type SkuGroup = {
  key: string;
  values: string[];
};

type QaItem = {
  question: string;
  answer: string;
};

const HIDDEN_SPEC_KEYS = new Set([
  "skus",
  "source",
  "source_json_path",
  "source_image_path",
  "image_path",
  "base_price",
  "category",
  "official_faq",
]);

export function ProductDetailPage() {
  const navigate = useNavigate();
  const { productId = "" } = useParams();
  const { userId, productSessionId } = useAppIdentity();
  const [product, setProduct] = useState<ProductDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [selectedSku, setSelectedSku] = useState<Record<string, string>>({});
  const [adding, setAdding] = useState(false);

  useEffect(() => {
    let ignore = false;
    setLoading(true);
    setError("");
    setSelectedSku({});
    getProduct(productId)
      .then((response) => {
        if (!ignore) setProduct(response);
      })
      .catch((err) => {
        if (!ignore) setError(err instanceof Error ? err.message : "商品不存在或加载失败");
      })
      .finally(() => {
        if (!ignore) setLoading(false);
      });
    return () => {
      ignore = true;
    };
  }, [productId]);

  const skuGroups = useMemo(() => extractSkuGroups(product?.specs), [product]);
  const specPairs = useMemo(() => flattenRecord(product?.specs ?? {}, HIDDEN_SPEC_KEYS), [product]);
  const attrPairs = useMemo(() => flattenRecord(product?.structured_attributes ?? {}, HIDDEN_SPEC_KEYS), [product]);
  const qaItems = useMemo(() => extractQaItems(product?.structured_attributes?.official_faq), [product]);
  const suitabilityPairs = useMemo<Array<[string, string]>>(() => {
    if (!product) return [];
    return [
      ["成分/材质", product.ingredients_or_material],
      ["适合人群/场景", product.suitable_for],
      ["不建议人群/避雷点", product.avoid_for],
      ["图片识别信息", product.image_caption],
    ].filter((entry): entry is [string, string] => Boolean(entry[1]?.trim()));
  }, [product]);
  const skuReady = skuGroups.every((group) => selectedSku[group.key]);

  async function addToCart() {
    if (!product || adding) return;
    if (skuGroups.length > 0 && !skuReady) {
      setNotice("请先选择完整规格");
      return;
    }
    setAdding(true);
    setNotice("");
    try {
      await reportAddToCart({
        userId,
        sessionId: productSessionId,
        productId: product.product_id,
        source: "product_detail",
        sku: selectedSku,
      });
      setNotice("已加入购物车");
    } catch {
      setNotice("加入购物车失败");
    } finally {
      setAdding(false);
    }
  }

  if (loading) {
    return (
      <section className="panel loading-panel">
        <Loader2 className="spin" size={22} />
        正在加载商品详情...
      </section>
    );
  }

  if (error || !product) {
    return (
      <section className="panel empty-page">
        <h1>商品加载失败</h1>
        <p>{error || "没有找到商品"}</p>
        <button className="button button-secondary" type="button" onClick={() => navigate(-1)}>
          <ArrowLeft size={16} />
          返回
        </button>
      </section>
    );
  }

  return (
    <div className="detail-layout">
      <section className="detail-media">
        <button className="button button-secondary back-button" type="button" onClick={() => navigate(-1)}>
          <ArrowLeft size={16} />
          返回
        </button>
        <img src={resolveAssetUrl(product.image_url)} alt={product.name} className="detail-image" />
      </section>

      <section className="detail-main">
        <div className="page-heading">
          <p className="eyebrow">{product.category}{product.sub_category ? ` / ${product.sub_category}` : ""}</p>
          <h1>{product.name}</h1>
          <p>{product.brand}</p>
        </div>

        <div className="detail-price-row">
          <strong>¥{formatPrice(product.price)}</strong>
          <span className="rating-chip large">
            <Star size={14} fill="currentColor" />
            {formatRating(product.rating)}
          </span>
          <span className="chip">{formatStock(product.stock)}</span>
          {product.sales != null ? <span className="chip">销量 {product.sales}</span> : null}
        </div>

        {product.tags.length > 0 ? (
          <div className="tag-row">
            {product.tags.slice(0, 10).map((tag) => (
              <span key={tag} className="tag-chip">{tag}</span>
            ))}
          </div>
        ) : null}

        <section className="detail-info-grid">
          <InfoTile label="品牌" value={product.brand} />
          <InfoTile label="分类" value={[product.category, product.sub_category].filter(Boolean).join(" / ")} />
          <InfoTile label="评分" value={formatRating(product.rating)} />
          <InfoTile label="库存" value={formatStock(product.stock)} />
        </section>

        {skuGroups.length > 0 ? (
          <section className="detail-section">
            <div className="section-title">
              <PackageCheck size={16} />
              <span>选择规格</span>
            </div>
            {skuGroups.map((group) => (
              <div key={group.key} className="sku-row">
                <div className="sku-key">{group.key}</div>
                <div className="sku-values">
                  {group.values.map((value) => (
                    <button
                      key={value}
                      type="button"
                      className={`sku-chip ${selectedSku[group.key] === value ? "active" : ""}`}
                      onClick={() => setSelectedSku((current) => ({ ...current, [group.key]: value }))}
                    >
                      {value}
                    </button>
                  ))}
                </div>
              </div>
            ))}
          </section>
        ) : null}

        <button className="button button-primary detail-cart-button" type="button" onClick={addToCart} disabled={adding}>
          {adding ? <Loader2 className="spin" size={18} /> : <ShoppingCart size={18} />}
          加入购物车
        </button>
        {notice ? <div className="notice-line">{notice}</div> : null}

        <TextSection
          icon={<BadgeCheck size={16} />}
          title="商品描述"
          content={product.description || "暂无商品描述。"}
        />

        {suitabilityPairs.length > 0 ? (
          <section className="detail-section">
            <div className="section-title">
              <Tags size={16} />
              <span>适用与避雷</span>
            </div>
            <KeyValueGrid pairs={suitabilityPairs} />
          </section>
        ) : null}

        {product.review_summary ? (
          <TextSection
            icon={<MessageSquareText size={16} />}
            title="用户评价摘要"
            content={product.review_summary}
          />
        ) : null}

        {specPairs.length > 0 ? (
          <section className="detail-section">
            <div className="section-title">
              <Info size={16} />
              <span>商品规格</span>
            </div>
            <KeyValueGrid pairs={specPairs} />
          </section>
        ) : null}

        {attrPairs.length > 0 ? (
          <section className="detail-section">
            <div className="section-title">
              <PackageCheck size={16} />
              <span>结构化属性</span>
            </div>
            <KeyValueGrid pairs={attrPairs} />
          </section>
        ) : null}

        {qaItems.length > 0 ? (
          <section className="detail-section">
            <div className="section-title">
              <MessageSquareText size={16} />
              <span>常见问题</span>
            </div>
            <div className="qa-list">
              {qaItems.map((item, index) => (
                <div key={`${item.question}-${index}`} className="qa-item">
                  <strong>{item.question}</strong>
                  <p>{item.answer}</p>
                </div>
              ))}
            </div>
          </section>
        ) : null}
      </section>
    </div>
  );
}

function InfoTile({ label, value }: { label: string; value: string }) {
  return (
    <div className="info-tile">
      <span>{label}</span>
      <strong>{value || "-"}</strong>
    </div>
  );
}

function TextSection({ icon, title, content }: { icon: ReactNode; title: string; content: string }) {
  return (
    <section className="detail-section">
      <div className="section-title">
        {icon}
        <span>{title}</span>
      </div>
      <div className="prose-panel">{content}</div>
    </section>
  );
}

function KeyValueGrid({ pairs }: { pairs: Array<[string, string]> }) {
  if (pairs.length === 0) {
    return <div className="empty-hint">暂无可展示参数。</div>;
  }
  return (
    <div className="kv-grid wide">
      {pairs.map(([key, value]) => (
        <div key={key} className="kv-row">
          <div className="kv-key">{humanizeKey(key)}</div>
          <div className="kv-value">{value}</div>
        </div>
      ))}
    </div>
  );
}

function extractSkuGroups(specs: Record<string, unknown> | undefined): SkuGroup[] {
  const skus = specs?.skus;
  if (!Array.isArray(skus)) return [];
  const groups = new Map<string, Set<string>>();

  for (const sku of skus) {
    if (!sku || typeof sku !== "object") continue;
    const properties = (sku as Record<string, unknown>).properties;
    if (!properties || typeof properties !== "object" || Array.isArray(properties)) continue;
    for (const [key, raw] of Object.entries(properties as Record<string, unknown>)) {
      const value = valueToText(raw);
      if (!key || !value) continue;
      if (!groups.has(key)) groups.set(key, new Set());
      groups.get(key)?.add(value);
    }
  }

  return Array.from(groups.entries()).map(([key, values]) => ({
    key: humanizeKey(key),
    values: Array.from(values),
  }));
}

function flattenRecord(obj: Record<string, unknown>, hidden: Set<string>, prefix = ""): Array<[string, string]> {
  const pairs: Array<[string, string]> = [];
  for (const [key, value] of Object.entries(obj)) {
    if (hidden.has(key)) continue;
    const label = prefix ? `${prefix}.${key}` : key;
    if (value && typeof value === "object" && !Array.isArray(value)) {
      const nested = flattenRecord(value as Record<string, unknown>, hidden, label);
      pairs.push(...nested);
      continue;
    }
    const text = valueToText(value);
    if (text) pairs.push([label, text]);
  }
  return pairs.slice(0, 28);
}

function extractQaItems(value: unknown): QaItem[] {
  if (!value) return [];
  if (Array.isArray(value)) {
    return value
      .map((item) => {
        if (typeof item === "string") return splitQuestionAnswer(item);
        if (!item || typeof item !== "object") return null;
        const record = item as Record<string, unknown>;
        return {
          question: valueToText(record.question ?? record.q ?? record.title),
          answer: valueToText(record.answer ?? record.a ?? record.content),
        };
      })
      .filter((item): item is QaItem => Boolean(item?.question && item.answer))
      .slice(0, 6);
  }
  if (typeof value === "object") {
    return Object.entries(value as Record<string, unknown>)
      .map(([question, answer]) => ({ question: humanizeKey(question), answer: valueToText(answer) }))
      .filter((item) => Boolean(item.question && item.answer))
      .slice(0, 6);
  }
  const item = splitQuestionAnswer(String(value));
  return item ? [item] : [];
}

function splitQuestionAnswer(value: string): QaItem | null {
  const text = value.trim();
  if (!text) return null;
  const separators = ["：", ":", "?", "？"];
  for (const separator of separators) {
    const index = text.indexOf(separator);
    if (index > 0 && index < text.length - 1) {
      return {
        question: text.slice(0, index + (separator === "?" || separator === "？" ? 1 : 0)).trim(),
        answer: text.slice(index + 1).trim(),
      };
    }
  }
  return { question: "说明", answer: text };
}

function valueToText(value: unknown): string {
  if (value == null) return "";
  if (typeof value === "string") return value.trim();
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  if (Array.isArray(value)) {
    return value
      .map((item) => valueToText(item))
      .filter(Boolean)
      .join("、")
      .slice(0, 220);
  }
  if (typeof value === "object") {
    return Object.entries(value as Record<string, unknown>)
      .map(([key, nested]) => {
        const text = valueToText(nested);
        return text ? `${humanizeKey(key)}：${text}` : "";
      })
      .filter(Boolean)
      .join("；")
      .slice(0, 220);
  }
  return "";
}

function humanizeKey(key: string) {
  return key.replace(/_/g, " ").replace(/\b\w/g, (char) => char.toUpperCase());
}

function formatPrice(value: string) {
  const number = Number(value);
  if (!Number.isFinite(number)) return value || "-";
  return number.toFixed(number >= 100 ? 0 : 2);
}

function formatRating(value: string) {
  const number = Number(value);
  if (!Number.isFinite(number)) return value || "-";
  return number.toFixed(2);
}

function formatStock(value?: number | null) {
  if (value == null) return "库存信息待确认";
  if (value <= 0) return "暂时缺货";
  if (value < 10) return `仅剩 ${value} 件`;
  return "库存充足";
}
