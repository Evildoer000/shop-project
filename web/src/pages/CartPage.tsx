import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Minus, Plus, ShoppingBag, Store, Trash2, Loader2 } from "lucide-react";
import { getCart, reportAddToCart, reportRemoveFromCart, resolveAssetUrl } from "../lib/api";
import { useAppIdentity } from "../lib/app-state";
import type { CartItem, CartResponse } from "../types";

export function CartPage() {
  const navigate = useNavigate();
  const { userId, cartSessionId } = useAppIdentity();
  const [cart, setCart] = useState<CartResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [updatingKey, setUpdatingKey] = useState("");

  useEffect(() => {
    loadCart();
  }, [userId]);

  async function loadCart() {
    setLoading(true);
    setError("");
    try {
      const response = await getCart(userId, "all");
      setCart(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : "加载购物车失败");
    } finally {
      setLoading(false);
    }
  }

  async function addOne(item: CartItem) {
    const key = itemKey(item);
    setUpdatingKey(key);
    setNotice("");
    try {
      await reportAddToCart({
        userId,
        sessionId: cartSessionId,
        productId: item.product_id,
        source: "cart_screen",
        sku: item.sku,
      });
      setNotice("已添加一件");
      await loadCart();
    } catch {
      setNotice("添加失败");
    } finally {
      setUpdatingKey("");
    }
  }

  async function removeOne(item: CartItem) {
    const key = itemKey(item);
    setUpdatingKey(key);
    setNotice("");
    try {
      await reportRemoveFromCart({
        userId,
        sessionId: cartSessionId,
        productId: item.product_id,
        source: "cart_screen",
        sku: item.sku,
      });
      setNotice("已减少一件");
      await loadCart();
    } catch {
      setNotice("移除失败");
    } finally {
      setUpdatingKey("");
    }
  }

  return (
    <div className="page-stack">
      <section className="page-heading page-heading-row">
        <div>
          <p className="eyebrow">Event Sourced Cart</p>
          <h1>购物车</h1>
          <p>购物车由行为事件反推，商品数量来自 cart_add、cart_remove 和 buy 的聚合。</p>
        </div>
        <button className="button button-secondary" type="button" onClick={loadCart} disabled={loading}>
          {loading ? <Loader2 className="spin" size={16} /> : <ShoppingBag size={16} />}
          刷新购物车
        </button>
      </section>

      {notice ? <div className="notice-line">{notice}</div> : null}
      {error ? <div className="error-line">{error}</div> : null}

      {loading ? (
        <section className="panel loading-panel">
          <Loader2 className="spin" size={22} />
          正在加载购物车...
        </section>
      ) : !cart || cart.items.length === 0 ? (
        <section className="panel empty-page">
          <ShoppingBag size={34} />
          <h2>购物车为空</h2>
          <p>可以从聊天推荐、商城推荐或商品详情页加入商品。</p>
          <button className="button button-primary" type="button" onClick={() => navigate("/mall")}>
            <Store size={16} />
            去商城逛逛
          </button>
        </section>
      ) : (
        <div className="cart-layout">
          <section className="cart-list">
            {cart.items.map((item) => (
              <article key={itemKey(item)} className="cart-item">
                <button className="cart-image-button" type="button" onClick={() => navigate(`/products/${item.product_id}`)} title="打开商品详情">
                  <img src={resolveAssetUrl(item.image_url)} alt={item.name} />
                </button>
                <div className="cart-item-body">
                  <h3>{item.name}</h3>
                  <div className="product-meta">
                    <span>{item.brand}</span>
                    <span>·</span>
                    <span>{item.category}</span>
                    {item.sub_category ? <><span>·</span><span>{item.sub_category}</span></> : null}
                  </div>
                  <div className="tag-row">
                    {Object.entries(item.sku).map(([key, value]) => (
                      <span key={key} className="tag-chip">{key}: {value}</span>
                    ))}
                  </div>
                  <strong className="product-price">¥{formatPrice(item.price)}</strong>
                </div>
                <div className="cart-quantity">
                  <button className="icon-button" type="button" onClick={() => removeOne(item)} disabled={updatingKey === itemKey(item)} title="减少一件">
                    {item.quantity <= 1 ? <Trash2 size={16} /> : <Minus size={16} />}
                  </button>
                  <span>{item.quantity}</span>
                  <button className="icon-button" type="button" onClick={() => addOne(item)} disabled={updatingKey === itemKey(item)} title="添加一件">
                    <Plus size={16} />
                  </button>
                </div>
              </article>
            ))}
          </section>

          <aside className="panel cart-summary">
            <div className="panel-title">
              <ShoppingBag size={16} />
              <span>合计</span>
            </div>
            <div className="summary-row">
              <span>商品数量</span>
              <strong>{cart.total_quantity}</strong>
            </div>
            <div className="summary-row">
              <span>总价</span>
              <strong>¥{formatPrice(cart.total_price)}</strong>
            </div>
            <button className="button button-primary button-block" type="button" disabled>
              下单演示
            </button>
          </aside>
        </div>
      )}
    </div>
  );
}

function itemKey(item: CartItem) {
  return `${item.product_id}:${JSON.stringify(item.sku ?? {})}`;
}

function formatPrice(value: number) {
  return Number.isFinite(value) ? value.toFixed(value >= 100 ? 0 : 2) : "-";
}
