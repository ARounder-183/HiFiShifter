# HiFiShifter 前端

本目录是 [HiFiShifter](../README.md) 的前端（React 19 + TypeScript + Vite），通过 Tauri 与 Rust 内核（`../backend/`）通信；同一套代码同时构建独立 App 与 Windows VST3/ARA 插件形态的界面。项目介绍、功能与使用方式见[根 README](../README.md)，本文只讲前端工程本身。

## 目录结构

```
src/
├── app/                # Redux store 与根选择器
├── assets/             # 静态资源
├── components/         # 业务表面（layout=轨道/参数/对话框，dock=停靠系统）
├── contexts/           # React Context（音高分析等）
├── features/           # Redux slices 与领域逻辑（session/dock/keybindings/…）
├── hooks/              # 跨表面共享的 React hooks
├── i18n/               # 5 语系词典（en-US/zh-CN/zh-TW/ja-JP/ko-KR）与完整性门禁
├── sdk/                # 第三方扩展 API（@hs/sdk）
├── services/           # Tauri invoke / 宿主事件 / 各域 API 封装
├── theme/              # 主题（Radix + 自定义 --qt-* 令牌）
├── types/              # 与后端载荷对应的全局类型
├── ui/                 # 无业务的原语（Dialog/Menu/NumberField/…）
├── utils/              # 工具与纯逻辑（吸附、mipmap 缓存、时间轴…）
└── waveform/           # 波形渲染（sceneBuilder/geometry/surfaceRenderer）
```

## 常用命令

```bash
npm install
npm run dev            # 开发服务器（Tauri 壳另由 backend 侧启动）
npm run typecheck      # tsc -b + SDK 声明检查（真正的类型门禁，见根 tsconfig.json 注释）
npm run test           # vitest run（全部测试，含构建门禁 buildGates）
npm run bench          # 波形/几何等性能基准
npm run lint           # eslint .
npm run lint:budget    # 同上，但限制警告数上限（见 package.json）
npm run format:check   # prettier 校验；format 为写入
npm run build          # 类型检查 + vite 生产构建
```

## 构建入口

`vite.config.ts` 定义多个 HTML 入口：`index.html`（主窗口）、`detached.html`（拆离窗口）、`plugin.html`（VST3/ARA 插件形态）。`waveform-test.html` 是开发期波形调试页，**默认不进生产构建**；需要打包它时设 `VITE_INCLUDE_TEST_PAGES=1`。

## 工程约定（有门禁把守的部分）

- **i18n**：新增 key 必须同步加入全部 5 个语言文件（`src/i18n/*.ts`），`catalogIntegrity.test.ts` 与 `keyReferenceIntegrity.test.ts` 会红。
- **测试环境**：`vitest.config.ts` 是中央配置，默认 `environment: "node"`；需要 DOM 的测试文件在文件头写 `// @vitest-environment jsdom` pragma。
- **Lint / 格式**：ESLint 警告走预算制（`lint:budget`）；提交前保持 `prettier --check` 零漂移。
- **对话框 / 菜单**：统一走 `src/ui/Dialog.tsx`（AppDialog）与 `src/ui/Menu.tsx`（AppContextMenu），不要手写模态壳（`interactionCoverage.test.ts` 会拦）。

## 相关文档

- 根 [README](../README.md) 与 [使用手册](../docs/i18n/USERMANUAL.md)
- `../docs/plans/` 下的审查与修复方案
