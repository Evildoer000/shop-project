import { ArrowRight, BadgePercent, Plus, ShoppingCart, Star } from "lucide-react";
import { resolveAssetUrl } from "../lib/api";
import type { ProductCard as ProductCardType } from "../types";

type ProductCardProps = {
  product: ProductCardType;
  onOpen?: () => void;
  onAdd?: () => void;
  compact?: boolean;
  showScore?: boolean;
};

export function ProductCard({ product, onOpen, onAdd, compact = false, showScore = false }: ProductCardProps) {
  const imageUrl = resolveAssetUrl(product.image_url);

  return (
    <article className={`product-card ${compact ? "product-card-compact" : ""}`}>
      <button className="product-image-wrap" onClick={onOpen} type="button" title="打开详情">
        <img src={imageUrl} alt={product.name} className="product-image" loading="lazy" />
        {showScore && typeof product.score === "number" ? (
          <span className="score-badge">
            <BadgePercent size={12} />
            {product.score.toFixed(3)}
          </span>
        ) : null}
      </button>

      <div className="product-body">
        <div className="product-title-row">
          <h3 className="product-title" title={product.name}>{product.name}</h3>
        </div>

        <div className="product-meta">
          <span>{product.brand}</span>
          <span>·</span>
          <span>{product.category}</span>
          {product.sub_category ? <><span>·</span><span>{product.sub_category}</span></> : null}
        </div>

        <div className="product-price-row">
          <strong className="product-price">¥{formatPrice(product.price)}</strong>
          <span className="rating-chip">
            <Star size={12} fill="currentColor" />
            {product.rating.toFixed(2)}
          </span>
        </div>

        {product.reason ? <div className="product-reason">{product.reason}</div> : null}

        <div className="tag-row">
          {product.tags.slice(0, 3).map((tag) => (
            <span key={tag} className="tag-chip">{tag}</span>
          ))}
        </div>

        <div className="product-actions">
          <button type="button" className="button button-ghost" onClick={onOpen}>
            <ArrowRight size={16} />
            详情
          </button>
          <button type="button" className="button button-primary" onClick={onAdd}>
            <ShoppingCart size={16} />
            加购
          </button>
        </div>
      </div>
    </article>
  );
}

function formatPrice(value: number) {
  return Number.isFinite(value) ? value.toFixed(value >= 100 ? 0 : 2) : "-";
}
