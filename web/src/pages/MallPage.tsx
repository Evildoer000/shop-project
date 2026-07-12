import { FormEvent, useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Grid3X3, Loader2, RefreshCw, Search, SlidersHorizontal, TrendingUp } from "lucide-react";
import { ProductCard } from "../components/ProductCard";
import {
  getProductCategories,
  getProducts,
  getRecommendations,
  reportAddToCart,
  reportEvent,
  reportMallOpen,
} from "../lib/api";
import { useAppIdentity } from "../lib/app-state";
import type { ProductCard as ProductCardType, ProductCategorySummary } from "../types";

type MallMode = "recommend" | "catalog";

const PAGE_SIZE = 24;

export function MallPage() {
  const navigate = useNavigate();
  const { userId, mallSessionId } = useAppIdentity();
  const [mode, setMode] = useState<MallMode>("recommend");
  const [products, setProducts] = useState<ProductCardType[]>([]);
  const [stage, setStage] = useState("");
  const [totalEvents, setTotalEvents] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [categories, setCategories] = useState<ProductCategorySummary[]>([]);
  const [selectedCategory, setSelectedCategory] = useState("");
  const [selectedSubCategory, setSelectedSubCategory] = useState("");
  const [searchText, setSearchText] = useState("");
  const [query, setQuery] = useState("");
  const [sort, setSort] = useState("popular");
  const [catalogPage, setCatalogPage] = useState(1);
  const [catalogTotal, setCatalogTotal] = useState(0);
  const impressedRef = useRef<Set<string>>(new Set());

  const selectedCategoryInfo = useMemo(
    () => categories.find((category) => category.name === selectedCategory),
    [categories, selectedCategory],
  );
  const hasMoreCatalog = mode === "catalog" && products.length < catalogTotal;

  useEffect(() => {
    impressedRef.current.clear();
  }, [userId, mode]);

  useEffect(() => {
    loadRecommendations();
  }, [userId]);

  useEffect(() => {
    let ignore = false;
    getProductCategories()
      .then((response) => {
        if (!ignore) setCategories(response.categories);
      })
      .catch(() => {
        if (!ignore) setCategories([]);
      });
    return () => {
      ignore = true;
    };
  }, []);

  useEffect(() => {
    if (mode === "catalog") {
      void loadCatalog(1, false);
    }
  }, [mode, selectedCategory, selectedSubCategory, query, sort]);

  useEffect(() => {
    products.slice(0, 12).forEach((product, index) => {
      const key = `${mode}:${product.product_id}`;
      if (impressedRef.current.has(key)) return;
      impressedRef.current.add(key);
      reportEvent({
        user_id: userId,
        session_id: mallSessionId,
        event_type: "impression",
        product_id: product.product_id,
        position: index,
        context: {
          from: mode === "catalog" ? "catalog" : "mall",
          page: mode,
          stage,
          brand: product.brand,
          category: product.category,
        },
      }).catch(() => undefined);
    });
  }, [products, stage, userId, mallSessionId, mode]);

  async function loadRecommendations() {
    setMode("recommend");
    setLoading(true);
    setError("");
    setNotice("");
    try {
      const response = await getRecommendations(userId, 24);
      setProducts(response.products);
      setStage(response.stage);
      setTotalEvents(response.total_events);
      setCatalogTotal(0);
    } catch (err) {
      setError(err instanceof Error ? err.message : "加载推荐失败");
    } finally {
      setLoading(false);
    }
  }

  async function loadCatalog(page: number, append: boolean) {
    setLoading(true);
    setError("");
    setNotice("");
    try {
      const response = await getProducts({
        category: selectedCategory,
        subCategory: selectedSubCategory,
        q: query,
        page,
        pageSize: PAGE_SIZE,
        sort,
      });
      setProducts((current) => (append ? [...current, ...response.products] : response.products));
      setCatalogPage(response.page);
      setCatalogTotal(response.total);
      setStage("catalog");
      setTotalEvents(response.total);
    } catch (err) {
      setError(err instanceof Error ? err.message : "加载商品失败");
    } finally {
      setLoading(false);
    }
  }

  function submitSearch(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setQuery(searchText.trim());
    setMode("catalog");
  }

  function switchToCatalog() {
    setMode("catalog");
  }

  function selectCategory(category: string) {
    setSelectedCategory(category);
    setSelectedSubCategory("");
    setMode("catalog");
  }

  function openProduct(product: ProductCardType, position: number) {
    reportMallOpen({
      userId,
      sessionId: mallSessionId,
      productId: product.product_id,
      position,
      brand: product.brand,
      category: product.category,
    }).catch(() => undefined);
    navigate(`/products/${product.product_id}`);
  }

  async function addToCart(product: ProductCardType) {
    try {
      await reportAddToCart({
        userId,
        sessionId: mallSessionId,
        productId: product.product_id,
        source: mode === "catalog" ? "catalog_card" : "mall_card",
      });
      setNotice("已加入购物车");
    } catch {
      setNotice("加入购物车失败");
    }
  }

  return (
    <div className="page-stack">
      <section className="page-heading page-heading-row">
        <div>
          <p className="eyebrow">{mode === "catalog" ? "Product Catalog" : "Recommendation Feed"}</p>
          <h1>{mode === "catalog" ? "全部商品" : "商城推荐"}</h1>
          <p>
            {mode === "catalog"
              ? "按分类、关键词和排序浏览完整商品库，点击卡片查看商品详情。"
              : "基于用户行为事件、曝光抑制和多样性重排生成推荐流。"}
          </p>
        </div>
        <div className="segmented-control">
          <button
            className={mode === "recommend" ? "active" : ""}
            type="button"
            onClick={loadRecommendations}
          >
            <TrendingUp size={16} />
            推荐
          </button>
          <button
            className={mode === "catalog" ? "active" : ""}
            type="button"
            onClick={switchToCatalog}
          >
            <Grid3X3 size={16} />
            全部商品
          </button>
        </div>
      </section>

      <section className="metrics-row">
        <div className="metric-tile">
          <TrendingUp size={18} />
          <div>
            <span>{mode === "catalog" ? "浏览模式" : "推荐阶段"}</span>
            <strong>{mode === "catalog" ? "catalog" : stage || "-"}</strong>
          </div>
        </div>
        <div className="metric-tile">
          <SlidersHorizontal size={18} />
          <div>
            <span>{mode === "catalog" ? "匹配商品" : "累计行为"}</span>
            <strong>{mode === "catalog" ? catalogTotal : totalEvents}</strong>
          </div>
        </div>
        <div className="metric-tile">
          <RefreshCw size={18} />
          <div>
            <span>本次返回</span>
            <strong>{products.length}</strong>
          </div>
        </div>
      </section>

      {mode === "catalog" ? (
        <section className="catalog-layout">
          <aside className="panel category-panel">
            <div className="panel-title">商品分类</div>
            <div className="category-list">
              <button
                className={`category-item ${selectedCategory === "" ? "active" : ""}`}
                type="button"
                onClick={() => selectCategory("")}
              >
                <span>全部分类</span>
                <strong>{categories.reduce((sum, item) => sum + item.count, 0)}</strong>
              </button>
              {categories.map((category) => (
                <button
                  key={category.name}
                  className={`category-item ${selectedCategory === category.name ? "active" : ""}`}
                  type="button"
                  onClick={() => selectCategory(category.name)}
                >
                  <span>{category.name}</span>
                  <strong>{category.count}</strong>
                </button>
              ))}
            </div>
          </aside>

          <div className="catalog-main">
            <section className="panel catalog-toolbar">
              <form className="catalog-search" onSubmit={submitSearch}>
                <Search size={16} />
                <input
                  value={searchText}
                  onChange={(event) => setSearchText(event.target.value)}
                  placeholder="搜索商品名、品牌、标签或描述"
                />
                <button className="button button-primary" type="submit">搜索</button>
              </form>

              <div className="toolbar-row">
                <div className="sub-category-row">
                  <button
                    className={`filter-chip ${selectedSubCategory === "" ? "active" : ""}`}
                    type="button"
                    onClick={() => setSelectedSubCategory("")}
                  >
                    全部子类
                  </button>
                  {selectedCategoryInfo?.sub_categories.map((subCategory) => (
                    <button
                      key={subCategory.name}
                      className={`filter-chip ${selectedSubCategory === subCategory.name ? "active" : ""}`}
                      type="button"
                      onClick={() => setSelectedSubCategory(subCategory.name)}
                    >
                      {subCategory.name} · {subCategory.count}
                    </button>
                  ))}
                </div>

                <select className="sort-select" value={sort} onChange={(event) => setSort(event.target.value)}>
                  <option value="popular">销量优先</option>
                  <option value="rating">评分优先</option>
                  <option value="price_asc">价格从低到高</option>
                  <option value="price_desc">价格从高到低</option>
                  <option value="newest">最近更新</option>
                </select>
              </div>
            </section>

            {renderProductArea()}
          </div>
        </section>
      ) : (
        renderProductArea()
      )}
    </div>
  );

  function renderProductArea() {
    return (
      <>
        {notice ? <div className="notice-line">{notice}</div> : null}
        {error ? <div className="error-line">{error}</div> : null}

        {loading && products.length === 0 ? (
          <section className="panel loading-panel">
            <Loader2 className="spin" size={22} />
            {mode === "catalog" ? "正在加载商品..." : "正在加载推荐商品..."}
          </section>
        ) : products.length === 0 ? (
          <section className="panel empty-page">
            <h2>没有找到商品</h2>
            <p>可以换一个分类或关键词再试。</p>
          </section>
        ) : (
          <>
            <section className="product-grid">
              {products.map((product, index) => (
                <ProductCard
                  key={`${mode}-${product.product_id}-${index}`}
                  product={product}
                  showScore={mode === "recommend"}
                  onOpen={() => openProduct(product, index)}
                  onAdd={() => addToCart(product)}
                />
              ))}
            </section>
            {mode === "catalog" && hasMoreCatalog ? (
              <div className="load-more-row">
                <button
                  className="button button-secondary"
                  type="button"
                  onClick={() => loadCatalog(catalogPage + 1, true)}
                  disabled={loading}
                >
                  {loading ? <Loader2 className="spin" size={16} /> : <RefreshCw size={16} />}
                  加载更多
                </button>
              </div>
            ) : null}
          </>
        )}
      </>
    );
  }
}
