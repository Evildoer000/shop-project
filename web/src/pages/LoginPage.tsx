import { FormEvent, useState } from "react";
import { Loader2, LockKeyhole, LogIn, Phone, ShieldCheck, Wifi, WifiOff } from "lucide-react";
import { useAppIdentity } from "../lib/app-state";
import type { HealthStatus } from "../types";

type LoginPageProps = {
  health: HealthStatus | null;
};

const demoPhones = ["13800000001", "13800000002", "13900000003"];

export function LoginPage({ health }: LoginPageProps) {
  const { login } = useAppIdentity();
  const [phone, setPhone] = useState(demoPhones[0]);
  const [password, setPassword] = useState("88888");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const healthy = health?.status === "ok";

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (loading) return;
    setLoading(true);
    setError("");
    try {
      await login(phone, password);
    } catch (err) {
      setError(err instanceof Error ? err.message : "登录失败");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="login-page">
      <section className="login-panel">
        <div className="login-brand">
          <div className="brand-mark">S</div>
          <div>
            <h1>Shop Agent 导购助手</h1>
            <p>使用手机号切换不同用户画像、推荐和购物车。</p>
          </div>
        </div>

        <div className={`status-pill ${healthy ? "status-ok" : "status-bad"}`}>
          {healthy ? <Wifi size={14} /> : <WifiOff size={14} />}
          <span>{healthy ? "后端在线" : "后端未连通"}</span>
        </div>

        <form className="login-form" onSubmit={submit}>
          <label className="login-field">
            <span>手机号</span>
            <div className="login-input-wrap">
              <Phone size={17} />
              <input
                value={phone}
                onChange={(event) => setPhone(event.target.value)}
                placeholder="请输入手机号"
                autoComplete="tel"
              />
            </div>
          </label>

          <label className="login-field">
            <span>密码</span>
            <div className="login-input-wrap">
              <LockKeyhole size={17} />
              <input
                value={password}
                onChange={(event) => setPassword(event.target.value)}
                placeholder="默认密码 88888"
                type="password"
                autoComplete="current-password"
              />
            </div>
          </label>

          <div className="demo-phone-row">
            {demoPhones.map((item) => (
              <button key={item} type="button" className="sku-chip" onClick={() => setPhone(item)}>
                {item}
              </button>
            ))}
          </div>

          {error ? <div className="error-line">{error}</div> : null}

          <button className="button button-primary button-block login-submit" type="submit" disabled={loading}>
            {loading ? <Loader2 className="spin" size={18} /> : <LogIn size={18} />}
            登录
          </button>
        </form>

        <div className="login-note">
          <ShieldCheck size={16} />
          <span>当前是演示登录：密码固定为 88888；手机号会映射为稳定 user_id。</span>
        </div>
      </section>
    </main>
  );
}
