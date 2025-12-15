# Magic Conch Frontend (AI Spirit Pet)

## 项目简介

"AI灵宠" (Magic Conch) 是一个基于 Vue 3 + Vite 开发的虚拟桌宠前端项目，旨在为用户打造专属的数字伴侣，提供跨次元的舞台互动与温馨陪伴体验。用户可以与不同的 AI 模型（如桜小路ルナ、小倉朝日）进行文本和语音对话，享受沉浸式的交互乐趣。

## 主要功能

*   **多模型对话**：支持与多个 AI 模型进行实时对话，支持流式响应。
*   **语音交互**：集成语音合成 (TTS) 功能，让 AI 角色“开口说话”。
*   **实时翻译**：内置百度翻译功能，支持中、日、英多语言互译，提供悬停翻译气泡，方便理解。
*   **自动转发**：支持消息自动转发功能，当模型输出中提及其他角色时，自动触发对应角色的回应，实现多角色联动。
*   **个性化设置**：
    *   **主题切换**：支持深色/浅色/跟随系统主题。
    *   **语言切换**：支持中文、英文、日文界面语言。
    *   **模型管理**：可自定义模型头像、名称，并针对每个模型单独配置语音、翻译等功能。
*   **沉浸式 UI**：精美的气泡背景动画、角色头像、流式打字机效果，营造灵动的交互氛围。

## 技术栈

*   **前端框架**: Vue 3 (Composition API)
*   **构建工具**: Vite
*   **UI 组件库**: Element Plus
*   **状态管理**: Pinia
*   **路由管理**: Vue Router
*   **HTTP 客户端**: Axios
*   **国际化**: 自研 i18n 简单实现
*   **CSS 预处理**: 原生 CSS Variables + Scoped CSS

## 安装与运行

### 前置要求

*   Node.js (推荐 v16+)
*   npm 或 yarn

### 步骤

1.  **克隆项目**

    ```bash
    git clone https://github.com/your-repo/magic-conch-frontend.git
    cd magic-conch-frontend
    ```

2.  **安装依赖**

    ```bash
    npm install
    ```

3.  **配置环境变量**

    如果需要配置后端 API 地址，请修改 `src/api/config.js` 中的 `BASE_URL`。
    
    *注意：百度翻译 API Key 等敏感信息请自行申请并在 `src/utils/baiduTranslate.js` 中配置，或使用环境变量注入（推荐）。*

4.  **启动开发服务器**

    ```bash
    npm run dev
    ```

    访问 `http://localhost:5173` (或终端显示的端口)。

5.  **构建生产版本**

    ```bash
    npm run build
    ```

## 主要目录结构

```
src/
├── api/                # API 接口配置
├── assets/             # 静态资源 (图片, CSS)
├── components/         # 公共组件
├── router/             # 路由配置
├── stores/             # Pinia 状态管理
├── utils/              # 工具函数 (请求, 翻译, i18n)
├── views/              # 页面视图
│   ├── content/        # 主要内容区 (聊天, 模型)
│   ├── settings/       # 设置页面
│   ├── profile/        # 关于页面
│   └── LoginPage.vue   # 登录/欢迎页
└── App.vue             # 根组件
```

## 核心功能实现详解

### 1. 聊天模块 (`src/views/content/chat/ChatPage.vue`)

聊天页面是应用的核心交互区，实现了多模型并发对话、消息流式处理、自动转发等复杂逻辑。

*   **消息发送 (`sendMessage`)**:
    *   用户输入文本后，系统首先创建一个 `user` 角色的消息对象推入 `currentConversation.messages`。
    *   遍历当前会话选中的所有模型 ID (`currentConversation.selectedModels`)，对每个模型调用 `handleModelResponse` 触发并发请求。
*   **模型响应处理 (`handleModelResponse`)**:
    *   创建一个 `assistant` 角色的占位消息，状态设为 `loading`。
    *   调用 `fetchModelReply` 获取后端 API 响应（包含文本、TTS 音频 URL、自动翻译结果）。
    *   响应返回后，更新消息内容，停止 loading 状态。
    *   **自动转发逻辑**: 若开启了全局自动转发 (`appStore.autoForward`)，则使用正则表达式检测响应文本中是否包含其他模型名称（如“桜小路ルナ”、“小倉朝日”）。若匹配且非当前说话模型，则递归调用 `handleModelResponse`，将该消息作为 Prompt 转发给被提及的模型，实现模型间的互动。
*   **翻译功能 (`translateMessage`)**:
    *   用户悬停翻译按钮时触发 `el-popover` 的 `show` 事件。
    *   调用 `translateText` (封装自 `baiduTranslate.js`)。
    *   **目标语言判定**: 根据全局语言设置 (`appStore.language`) 动态决定翻译的目标语言（App 为中文则译为中文，App 为日文则译为日文）。
*   **音频播放 (`playAudio`)**:
    *   检查消息对象中是否包含 `audioUrl`。
    *   使用原生 `new Audio(url).play()` 播放 TTS 生成的语音。

### 2. 全局状态管理 (`src/stores/appStore.js`)

使用 Pinia 管理应用的全局配置，确保状态在页面间共享并持久化。

*   **主题管理 (`applyTheme`)**:
    *   支持 `light`, `dark`, `system` 三种模式。
    *   通过操作 `document.documentElement.classList` 添加或移除 `dark` 类来实现 CSS 变量切换。
    *   监听 `window.matchMedia('(prefers-color-scheme: dark)')` 实现系统主题跟随。
*   **持久化**:
    *   利用 Vue `watch` 监听状态变化，实时写入 `localStorage` (`app-theme`, `app-lang`, `app-auto-forward`)，页面刷新后自动还原。

### 3. 翻译服务 (`src/utils/baiduTranslate.js`)

封装了百度通用文本翻译 API (LLM 版)，解决了跨域和签名问题。

*   **API 封装 (`translateTextBaidu`)**:
    *   构造请求 Payload，包含 `appid`, `from`, `to`, `q` (文本)。
    *   使用 `fetch` 发送 POST 请求。
    *   **鉴权**: 使用 `Bearer Token` 方式进行鉴权 (`Authorization` header)。
*   **跨域代理 (Vite Proxy)**:
    *   前端请求 `/baidu-api/ait/api/aiTextTranslate`。
    *   `vite.config.js` 配置 Proxy 将请求转发至 `https://fanyi-api.baidu.com`，绕过浏览器 CORS 限制。
    *   后端不存储 Key，Key 仅存在于前端构建配置或环境变量中。

### 4. 国际化 (i18n) (`src/utils/i18n.js`)

实现了一个轻量级的 i18n 钩子 `useI18n`。

*   **实现原理**:
    *   维护一个 `messages` 对象，包含 `zh`, `en`, `ja` 的翻译键值对。
    *   `t(path, args)` 函数根据 `appStore.language` 获取当前语言，解析路径（如 `chat.send`），并支持简单的参数插值 (`{name}`)。
    *   当 Store 中的语言变化时，组件重新渲染，界面语言即时切换。

### 5. 模型配置与数据持久化

*   **模型存储 (`ModelsPage.vue`)**:
    *   模型信息（名称、头像、功能开关）保存在 `localStorage` 的 `models-settings` 中。
    *   支持用户自定义上传头像（Base64 编码存储）。
*   **会话存储**:
    *   聊天记录保存在 `localStorage` 的 `chat-conversations` 中，包含完整的消息历史、选中模型状态。

## 贡献者

*   **Authored by**: Soulw1nd, firefly163
*   **Special Thanks**:
    *   annali07 (luna-sama)
    *   RVC-Boss (GPT-SoVITS)
    *   huangyf2013320506 (magic_conch)

## 许可证

MIT
