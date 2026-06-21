# 智购助手 · Android Client

Android 原生客户端，使用 **Kotlin + Jetpack Compose + OkHttp(SSE)**，对接 FastAPI 后端的
对话导购、商品详情、商城推荐和购物车接口。

## 功能闭环

- 对话页面（`ChatScreen`）：支持文字/图片输入，接收 SSE 事件并逐 token 渲染回复。
- Agent 进度条（`AgentUpdateStrip`）：展示 Planner 摘要、图片理解、证据校验等轻量进度。
- 商品卡片：从 SSE `product_cards` 事件解析，展示真实图片，点击进入商品详情页。
- 决策过程面板（`DecisionTracePanel`）：展示需求理解、检索摘要、证据反思和最终依据。
- 商品详情（`ProductDetailScreen`）：调用 `GET /api/products/{product_id}`，展示完整字段。
- 商城推荐页（`MallScreen`）：调用 `GET /api/recommendations` 展示非对话商品流。
- 购物车页（`CartScreen`）：展示已加入购物车的商品。
- 新会话按钮：重置 `session_id`。
- 提示词建议：空对话页提供三条复杂 query 示例，方便演示。

## 工程结构

```
client/android/
├── build.gradle.kts             # 顶层 Gradle 配置（AGP/Kotlin/Compose 插件版本）
├── settings.gradle.kts
├── gradle.properties
├── gradle/wrapper/              # Gradle Wrapper 配置（需 ./gradlew wrapper 生成 jar）
└── app/
    ├── build.gradle.kts         # 应用模块配置，BuildConfig.API_BASE_URL 注入
    └── src/main/
        ├── AndroidManifest.xml
        ├── res/                 # strings / themes / network_security_config 等
        └── java/com/bytedance/shopguide/
            ├── MainActivity.kt
            ├── ShopGuideApp.kt
            ├── data/            # ApiConfig / ChatStreamClient(SSE) / ProductApi
            ├── model/           # ProductCardDto / ProductDetailDto / DecisionTraceDto / ChatMessage
            └── ui/
                ├── chat/        # ChatScreen / ChatViewModel / MessageBubble / DecisionTracePanel / ChatInputBar
                ├── product/     # ProductDetailScreen / ProductDetailViewModel
                ├── nav/         # AppNav (Navigation Compose)
                └── theme/       # Material3 主题
```

## 与后端的接口契约

客户端在以下事件名上做了严格映射，对应 `server/app/domain/orchestrator.py` 中的输出：

| event              | 客户端处理                                              |
| ------------------ | --------------------------------------------------- |
| `agent_update`     | 追加 Planner / 图片理解 / Corrective 的用户可见进度文本 |
| `trace`            | 追加到当前消息的 `traceLogs`，在决策面板「Pipeline 阶段」中展示 |
| `decision_trace`   | 解析为 `DecisionTraceDto`，绑定到当前消息             |
| `token`            | 拼接到当前消息 `content` 上，实现逐字流式渲染          |
| `product_cards`    | 解析为 `ProductCardDto` 列表并挂载到消息              |
| `done`             | 流式状态结束                                           |
| `error`            | 写入消息错误态                                          |

请求体（`ChatStreamRequest`）：

```json
{
  "user_id":  "android_user",
  "session_id": "android_session_xxx",
  "message":  "我是油皮，预算 150 以内，推荐一款夏天用不闷的防晒",
  "image_id": null
}
```

## 后端地址配置

默认 `BuildConfig.API_BASE_URL = "http://10.0.2.2:8000/"`，即 Android 模拟器访问宿主机的回环地址。
要在真机或局域网设备上测试，可以在 `client/android/local.properties` 中加一行：

```properties
API_BASE_URL=http://192.168.31.100:8000/
```

或者运行时通过 Gradle property 传入：

```bash
./gradlew :app:assembleDebug -PAPI_BASE_URL=http://192.168.31.100:8000/
```

`res/xml/network_security_config.xml` 已经允许常见局域网网段使用明文 HTTP，仅用于本地开发。

## 启动方式

1. 启动后端（项目根目录）：
   ```bash
   docker compose up -d --build
   curl http://localhost:8000/health
   ```
2. 在 Android Studio 中：`File → Open... → client/android`。
3. 等待 Gradle 同步完成。如果是第一次拉取仓库，需要运行：
   ```bash
   ./gradlew wrapper
   ```
   以生成 `gradle/wrapper/gradle-wrapper.jar`（出于体积考虑没有提交到仓库）。
4. 选择 `app` 配置，在模拟器（或真机，已按上面配好 LAN 地址）上运行。

## 关键设计点

- **SSE 解析**：`ChatStreamClient` 直接消费 OkHttp `BufferedSource`，遵循
  `event:`/`data:` + 空行 的标准 SSE 帧格式；`readTimeout = 0` 避免被空闲检测打断。
- **状态模型**：`ChatViewModel` 暴露单一 `StateFlow<ChatUiState>`，每条助手消息持有
  自己的 `traceLogs` / `decisionTrace` / `products`，省去多个 LiveData。
- **防幻觉**：客户端只渲染来自 `product_cards` 事件的商品；详情页通过 `product_id` 回查后端，
  从源头杜绝端侧凭文本生成的伪商品。
- **可解释 UI**：`DecisionTracePanel` 仅展示后端给的摘要字段（需求理解 / 图片理解 / 检索摘要 /
  证据反思 / 最终依据），不展示模型思维链。

## 后续可扩展点

- 用户偏好编辑：`GET/PUT /api/user/profile` 可以挂到独立 Tab。
- 购物车事件可以继续扩展为订单确认、地址选择和支付前确认。
- 商城行为事件可以扩展为画像蒸馏和个性化排序特征。
