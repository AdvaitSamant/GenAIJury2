"""
Presentation styling system for GenAI Jury 2.
Visual theme: warm, editorial, Claude-like.
Palette:
- Canvas: #F5F3EE
- Cards: #FBFAF7
- Borders: #E4DFD5
- Body text: #2B2823
- Muted labels: #6B655B
- Accent: #C96442
- Dark mode support via CSS variables and prefers-color-scheme.
"""

# Palette definition constants
PALETTE = {
    "canvas": "#F5F3EE",
    "card": "#FBFAF7",
    "card_subtle": "#F1EEE7",
    "border": "#E4DFD5",
    "text_primary": "#2B2823",
    "text_secondary": "#6B655B",
    "text_muted": "#8C857B",
    "accent": "#C96442",
    "accent_tint": "#F6E7E0",
    "success": "#5E7D5B",
    "success_tint": "#EDF3ED",
    "warning": "#B7862B",
    "warning_tint": "#FAF4E8",
    "error": "#A4453A",
    "error_tint": "#FAECE9",
}

# SVG Line Icons for Domain Explorer (thin 1.5px stroke, no fill)
DOMAIN_ICONS = {
    "Cards": """<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="5" width="20" height="14" rx="2"/><line x1="2" y1="10" x2="22" y2="10"/></svg>""",
    "Transfers": """<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><line x1="17" y1="7" x2="7" y2="17"/><polyline points="17 17 7 17 7 7"/></svg>""",
    "Security": """<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>""",
    "Loans": """<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="1" x2="12" y2="23"/><path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/></svg>""",
    "Accounts": """<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>""",
    "Fees & Support": """<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>""",
    "Investments": """<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 7 13.5 15.5 8.5 10.5 2 17"/><polyline points="16 7 22 7 22 13"/></svg>""",
    "Cash & ATM": """<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="6" width="20" height="12" rx="2"/><circle cx="12" cy="12" r="2"/></svg>""",
}

EMPTY_STATE_ICON = """<svg class="empty-state-icon" width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>"""

# Zero-indentation CSS to guarantee markdown parser treats this strictly as a Type 1 HTML block
_CSS_BODY = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Inter:wght@400;500;600&family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;0,8..60,700;1,8..60,400&display=swap');

/* 1. Theme variables & Dark Mode */
:root,
[data-theme="light"] {
  --bg-canvas: #F5F3EE;
  --bg-card: #FBFAF7;
  --bg-card-subtle: #F1EEE7;
  --border-color: #E4DFD5;
  --border-subtle: #EDE9E1;
  --text-primary: #2B2823;
  --text-secondary: #6B655B;
  --text-muted: #8C857B;
  --accent: #C96442;
  --accent-tint: #F6E7E0;
  --accent-hover: #B85635;
  --success: #5E7D5B;
  --success-tint: #EDF3ED;
  --warning: #B7862B;
  --warning-tint: #FAF4E8;
  --error: #A4453A;
  --error-tint: #FAECE9;
  --font-serif: "Source Serif 4", Georgia, serif;
  --font-sans: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  --font-mono: "IBM Plex Mono", "JetBrains Mono", Menlo, monospace;
}

@media (prefers-color-scheme: dark) {
  :root {
    --bg-canvas: #1F1D1A;
    --bg-card: #2A2723;
    --bg-card-subtle: #332F2A;
    --border-color: #3D3831;
    --border-subtle: #322E27;
    --text-primary: #EDE8E1;
    --text-secondary: #A8A196;
    --text-muted: #7E776D;
    --accent: #C96442;
    --accent-tint: #3D261E;
    --accent-hover: #D97350;
    --success: #7DA67B;
    --success-tint: #232E22;
    --warning: #CCA03D;
    --warning-tint: #332A17;
    --error: #BA5449;
    --error-tint: #381E1C;
  }
}

[data-theme="dark"] {
  --bg-canvas: #1F1D1A;
  --bg-card: #2A2723;
  --bg-card-subtle: #332F2A;
  --border-color: #3D3831;
  --border-subtle: #322E27;
  --text-primary: #EDE8E1;
  --text-secondary: #A8A196;
  --text-muted: #7E776D;
  --accent: #C96442;
  --accent-tint: #3D261E;
  --accent-hover: #D97350;
  --success: #7DA67B;
  --success-tint: #232E22;
  --warning: #CCA03D;
  --warning-tint: #332A17;
  --error: #BA5449;
  --error-tint: #381E1C;
}

/* 2. Base resets and global layout */
html, body, [class*="css"] {
  font-family: var(--font-sans);
  color: var(--text-primary);
  background-color: var(--bg-canvas);
  font-size: 15px;
  line-height: 1.6;
}

.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > .main {
  background-color: var(--bg-canvas) !important;
  color: var(--text-primary) !important;
}

[data-testid="stHeader"] {
  background: transparent !important;
}

.block-container {
  max-width: 1440px;
  padding-top: 1.5rem;
  padding-bottom: 2.5rem;
}

h1, h2, h3, h4, h5, h6 {
  font-family: var(--font-serif) !important;
  color: var(--text-primary) !important;
  letter-spacing: -0.01em;
  font-weight: 600;
}

h1 { font-size: 30px !important; line-height: 1.25; }
h2 { font-size: 24px !important; line-height: 1.3; }
h3 { font-size: 20px !important; line-height: 1.35; }
h4 { font-size: 18px !important; line-height: 1.4; }
p, li { line-height: 1.6; font-size: 15px; color: var(--text-primary); }

/* 3. Sidebar: parchment, hairline divider, definition list */
[data-testid="stSidebar"] {
  background-color: var(--bg-canvas) !important;
  border-right: 1px solid var(--border-color) !important;
  box-shadow: none !important;
}

[data-testid="stSidebar"] * {
  color: var(--text-primary);
}

[data-testid="stSidebar"] .block-container {
  padding: 1.75rem 1.25rem 2.5rem;
}

[data-testid="stSidebar"] hr {
  border-color: var(--border-color) !important;
  margin: 1.5rem 0;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
  color: var(--text-secondary);
}

.hw-spec-list {
  margin: 0.5rem 0 0.8rem;
}
.hw-spec-row {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  padding: 7px 0;
  border-bottom: 1px solid var(--border-color);
  font-size: 13px;
}
.hw-spec-row:last-child {
  border-bottom: none;
}
.hw-label {
  color: var(--text-secondary);
  font-size: 13px;
}
.hw-val {
  font-family: var(--font-mono);
  color: var(--text-primary);
  font-weight: 500;
  text-align: right;
  font-size: 13px;
}
.vram-stat-row {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  padding: 8px 0 4px;
  font-size: 13px;
}
.vram-stat-val {
  font-family: var(--font-mono);
  font-size: 13px;
  font-weight: 500;
  color: var(--text-primary);
}
.vram-bar-track {
  height: 4px;
  background-color: var(--border-color);
  border-radius: 2px;
  overflow: hidden;
  margin: 4px 0 10px;
}
.vram-bar-fill {
  height: 100%;
  background-color: var(--accent);
  border-radius: 2px;
}

.section-eyebrow {
  font-size: 13px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  color: var(--text-secondary);
  margin: 0 0 4px;
}
.eyebrow-helper {
  font-size: 13px;
  color: var(--text-secondary);
  line-height: 1.45;
  margin: 0 0 12px;
}

/* 4. Sliders and Toggle Switch */
.slider-label-row {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  margin-top: 14px;
  margin-bottom: 2px;
}
.slider-title {
  font-size: 13px;
  font-weight: 500;
  color: var(--text-primary);
}
.slider-mono-val {
  font-family: var(--font-mono);
  font-size: 13px;
  font-weight: 500;
  color: var(--accent);
}

[data-testid="stSlider"] [role="slider"] {
  background-color: var(--accent) !important;
  border: 2px solid var(--bg-card) !important;
  box-shadow: 0 1px 3px rgba(43,40,35,0.15) !important;
  width: 15px !important;
  height: 15px !important;
}

[data-testid="stSlider"] div[data-baseweb="slider"] div[style*="background"] {
  background-color: var(--accent) !important;
}

[data-testid="stSlider"] [data-testid="stThumbValue"] {
  display: none !important;
}

[data-testid="stToggle"] {
  margin-top: 14px;
}
[data-testid="stToggle"] [aria-checked="true"],
[data-testid="stCheckbox"] input:checked ~ div,
div[data-baseweb="toggle"] [aria-checked="true"] {
  background-color: var(--accent) !important;
}
.toggle-helper {
  font-size: 12px;
  line-height: 1.4;
  color: var(--text-secondary);
  margin-top: 4px;
  margin-bottom: 12px;
}

/* 5. Plain Header */
.plain-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-end;
  gap: 20px;
  padding-bottom: 18px;
  border-bottom: 1px solid var(--border-color);
  margin-bottom: 24px;
}
.header-title {
  font-family: var(--font-serif) !important;
  font-size: 30px !important;
  font-weight: 600 !important;
  color: var(--text-primary) !important;
  margin: 0 !important;
  line-height: 1.25 !important;
  letter-spacing: -0.01em !important;
}
.header-subtitle {
  font-size: 15px;
  color: var(--text-secondary);
  margin: 6px 0 0 !important;
  line-height: 1.4;
}
.header-model-pill {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  border: 1px solid var(--border-color);
  background-color: var(--bg-card);
  border-radius: 999px;
  padding: 6px 12px;
  font-family: var(--font-mono);
  font-size: 12px;
  color: var(--text-secondary);
  white-space: nowrap;
}
.clay-dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background-color: var(--accent);
  display: inline-block;
}

/* 6. Text-based tab navigation */
[data-testid="stTabs"] [role="tablist"],
div[data-baseweb="tab-list"] {
  background: transparent !important;
  border: none !important;
  border-bottom: 1px solid var(--border-color) !important;
  padding: 0 0 2px 0 !important;
  gap: 24px !important;
  border-radius: 0 !important;
  margin-bottom: 24px !important;
}

[data-testid="stTabs"] [role="tab"],
button[data-baseweb="tab"] {
  background: transparent !important;
  border: none !important;
  border-bottom: 2px solid transparent !important;
  border-radius: 0 !important;
  color: var(--text-secondary) !important;
  font-family: var(--font-sans) !important;
  font-size: 15px !important;
  font-weight: 500 !important;
  padding: 8px 2px 10px 2px !important;
  margin: 0 !important;
  box-shadow: none !important;
  transition: color 140ms ease, border-color 140ms ease !important;
}

[data-testid="stTabs"] [role="tab"]:hover,
button[data-baseweb="tab"]:hover {
  color: var(--text-primary) !important;
  background: transparent !important;
}

[data-testid="stTabs"] [role="tab"][aria-selected="true"],
button[data-baseweb="tab"][aria-selected="true"] {
  color: var(--text-primary) !important;
  border-bottom: 2px solid var(--accent) !important;
  background: transparent !important;
  font-weight: 600 !important;
  box-shadow: none !important;
}

div[data-baseweb="tab-highlight"] {
  background-color: var(--accent) !important;
  height: 2px !important;
}

/* 7. Chat portal: Assistant prose & User card */
.chat-section-heading {
  font-family: var(--font-serif) !important;
  font-size: 20px !important;
  font-weight: 600 !important;
  color: var(--text-primary) !important;
  margin-bottom: 18px !important;
}

.assistant-msg-wrap {
  margin: 16px 0 20px;
}
.assistant-label {
  font-family: var(--font-serif);
  font-style: italic;
  font-size: 14px;
  color: var(--text-secondary);
  margin-bottom: 6px;
}
.assistant-prose {
  font-size: 15px;
  line-height: 1.6;
  color: var(--text-primary);
}
.assistant-prose p {
  margin: 6px 0;
}
.assistant-prose strong,
.assistant-prose b,
.action-text strong,
.action-text b,
.user-card strong,
.user-card b {
  font-weight: 600 !important;
  color: var(--text-primary) !important;
}
.assistant-prose a,
.action-text a,
.user-card a {
  color: var(--accent) !important;
  text-decoration: underline !important;
  text-underline-offset: 3px !important;
  font-weight: 500 !important;
  cursor: pointer !important;
  transition: opacity 140ms ease !important;
}
.assistant-prose a:hover,
.action-text a:hover,
.user-card a:hover {
  opacity: 0.8 !important;
  color: var(--accent) !important;
}

.user-msg-wrap {
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  margin: 16px 0 20px;
}
.user-label {
  font-size: 13px;
  color: var(--text-secondary);
  margin-bottom: 4px;
}
.user-card {
  background-color: var(--bg-card);
  border: 1px solid var(--border-color);
  border-radius: 12px;
  padding: 12px 16px;
  max-width: 82%;
  color: var(--text-primary);
  font-size: 15px;
  line-height: 1.55;
  box-shadow: 0 1px 2px rgba(43,40,35,0.03);
}

.action-list {
  margin: 12px 0;
  padding: 0;
  list-style: none;
}
.action-item {
  display: flex;
  gap: 12px;
  padding: 9px 0;
  border-bottom: 1px solid var(--border-color);
  font-size: 15px;
  line-height: 1.5;
}
.action-item:last-child {
  border-bottom: none;
}
.action-num {
  font-family: var(--font-serif);
  font-weight: 600;
  color: var(--accent);
  min-width: 20px;
}
.action-text {
  color: var(--text-primary);
  flex: 1;
}

[data-testid="stChatMessageAvatarAssistant"],
[data-testid="stChatMessageAvatarUser"] {
  display: none !important;
}
[data-testid="stChatMessage"] {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
  padding: 0 !important;
}

/* 8. Quick prompts & Chat composer input */
.quick-prompts-label {
  font-size: 12px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  color: var(--text-secondary);
  margin: 18px 0 6px 0;
}

[data-testid="stColumn"] .stButton > button {
  background-color: var(--bg-card) !important;
  border: 1px solid var(--border-color) !important;
  border-radius: 999px !important;
  color: var(--text-primary) !important;
  font-family: var(--font-sans) !important;
  font-size: 13px !important;
  font-weight: 500 !important;
  padding: 5px 14px !important;
  min-height: 32px !important;
  height: 32px !important;
  width: 100% !important;
  display: inline-flex !important;
  align-items: center !important;
  justify-content: center !important;
  box-shadow: 0 1px 2px rgba(43,40,35,0.03) !important;
  transition: border-color 140ms ease, background-color 140ms ease, color 140ms ease !important;
  white-space: nowrap !important;
}
[data-testid="stColumn"] .stButton > button:hover {
  border-color: var(--accent) !important;
  background-color: var(--accent-tint) !important;
  color: var(--accent) !important;
}

[data-testid="stChatInput"] {
  background: transparent !important;
  padding: 0 !important;
  margin-top: 14px;
}
[data-testid="stChatInput"] > div {
  background-color: var(--bg-card) !important;
  border: 1px solid var(--border-color) !important;
  border-radius: 14px !important;
  box-shadow: 0 1px 3px rgba(43,40,35,0.04) !important;
  padding: 4px 6px !important;
  transition: border-color 150ms ease !important;
}
[data-testid="stChatInput"] > div:focus-within {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 1px var(--accent) !important;
}
[data-testid="stChatInput"] textarea {
  background-color: transparent !important;
  color: var(--text-primary) !important;
  font-family: var(--font-sans) !important;
  font-size: 15px !important;
  line-height: 1.5 !important;
}
[data-testid="stChatInput"] textarea::placeholder {
  color: var(--text-secondary) !important;
  opacity: 0.85 !important;
}
[data-testid="stChatInput"] button {
  background-color: var(--accent) !important;
  border: none !important;
  border-radius: 50% !important;
  width: 32px !important;
  height: 32px !important;
  min-height: 32px !important;
  color: #ffffff !important;
  display: inline-flex !important;
  align-items: center !important;
  justify-content: center !important;
  transition: background-color 150ms ease !important;
}
[data-testid="stChatInput"] button:hover:not(:disabled) {
  background-color: var(--accent-hover) !important;
}
[data-testid="stChatInput"] button:disabled {
  background-color: var(--border-color) !important;
  color: var(--text-muted) !important;
  opacity: 0.5 !important;
}

/* 9. Real-Time Diagnostics Panel */
.empty-state-diag {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 56px 24px;
  text-align: center;
  background-color: var(--bg-card);
  border: 1px solid var(--border-color);
  border-radius: 8px;
  margin: 8px 0;
}
.empty-state-icon {
  stroke: var(--text-secondary);
  margin-bottom: 12px;
}
.empty-state-text {
  font-size: 14px;
  color: var(--text-secondary);
  margin: 0;
  max-width: 280px;
  line-height: 1.5;
}

.diag-panel-block {
  background-color: var(--bg-card);
  border: 1px solid var(--border-color);
  border-radius: 8px;
  padding: 16px;
  margin-bottom: 16px;
}
.intent-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 8px;
  margin: 6px 0 14px;
  flex-wrap: wrap;
}
.intent-code-chip {
  font-family: var(--font-mono);
  font-size: 13px;
  background-color: var(--bg-card-subtle);
  border: 1px solid var(--border-color);
  border-radius: 6px;
  padding: 3px 8px;
  color: var(--text-primary);
}
.badge-status {
  display: inline-flex;
  align-items: center;
  padding: 3px 8px;
  border-radius: 999px;
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.02em;
}
.badge-status.badge-success {
  background-color: var(--success-tint);
  color: var(--success);
  border: 1px solid var(--success);
}
.badge-status.badge-warning {
  background-color: var(--warning-tint);
  color: var(--warning);
  border: 1px solid var(--warning);
}
.badge-status.badge-error {
  background-color: var(--error-tint);
  color: var(--error);
  border: 1px solid var(--error);
}

.confidence-serif-val {
  font-family: var(--font-serif);
  font-size: 26px;
  font-weight: 600;
  color: var(--text-primary);
  line-height: 1.2;
  margin: 4px 0 8px;
}
.confidence-bar-track {
  height: 4px;
  background-color: var(--border-color);
  border-radius: 2px;
  overflow: hidden;
}
.confidence-bar-fill {
  height: 100%;
  background-color: var(--accent);
  border-radius: 2px;
}

[data-testid="stExpander"] {
  background-color: var(--bg-card) !important;
  border: 1px solid var(--border-color) !important;
  border-radius: 8px !important;
  box-shadow: none !important;
  margin: 12px 0 !important;
}
[data-testid="stExpander"] summary {
  font-family: var(--font-sans) !important;
  font-weight: 500 !important;
  font-size: 14px !important;
  color: var(--text-primary) !important;
}

.proactive-card {
  background-color: var(--bg-card);
  border: 1px solid var(--border-color);
  border-left: 2px solid var(--accent);
  border-radius: 6px;
  padding: 12px 14px;
  margin: 8px 0;
  transition: background-color 150ms ease;
}
.proactive-card:hover {
  background-color: var(--bg-card-subtle);
}
.proactive-title {
  font-family: var(--font-sans);
  font-weight: 600;
  font-size: 14px;
  color: var(--text-primary);
}
.proactive-desc {
  font-size: 13px;
  color: var(--text-secondary);
  margin: 3px 0 8px;
  line-height: 1.4;
}
.proactive-chips {
  display: flex;
  gap: 8px;
}
.mono-chip {
  font-family: var(--font-mono);
  font-size: 11px;
  background-color: var(--bg-card-subtle);
  border: 1px solid var(--border-color);
  border-radius: 4px;
  padding: 2px 6px;
  color: var(--text-secondary);
}
.mono-chip strong {
  color: var(--text-primary);
  font-weight: 500;
}

.diag-split-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 10px;
  margin-top: 8px;
}
.diag-subcard {
  background-color: var(--bg-card);
  border: 1px solid var(--border-color);
  border-radius: 6px;
  padding: 10px 12px;
}
.subcard-label {
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  font-weight: 600;
  color: var(--text-secondary);
}
.grounded-val-serif {
  font-family: var(--font-serif);
  font-size: 20px;
  font-weight: 600;
  color: var(--text-primary);
  margin-top: 4px;
}

/* 10. Metrics Tab: 6 KPI figures in a row */
.kpi-row-grid {
  display: grid;
  grid-template-columns: repeat(6, 1fr);
  gap: 12px;
  margin: 16px 0 28px;
}
.kpi-stat-card {
  background-color: var(--bg-card);
  border: 1px solid var(--border-color);
  border-radius: 8px;
  padding: 14px 12px;
}
.kpi-label-eyebrow {
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  font-weight: 600;
  color: var(--text-secondary);
}
.kpi-serif-val {
  font-family: var(--font-serif);
  font-size: 26px;
  font-weight: 600;
  color: var(--text-primary);
  line-height: 1.2;
  margin: 6px 0 4px;
}
.kpi-desc-quiet {
  font-size: 12px;
  color: var(--text-muted);
}

/* Container cards with border */
[data-testid="stVerticalBlockBorderWrapper"] {
  background-color: var(--bg-card) !important;
  border: 1px solid var(--border-color) !important;
  border-radius: 8px !important;
  box-shadow: 0 1px 2px rgba(43,40,35,0.03) !important;
  margin-bottom: 16px !important;
}

.chart-header-block {
  margin-bottom: 12px;
}
.chart-serif-title {
  font-family: var(--font-serif) !important;
  font-size: 18px !important;
  font-weight: 600 !important;
  color: var(--text-primary) !important;
  margin: 0 0 6px 0 !important;
}
.chart-explanation-text {
  font-size: 13px !important;
  line-height: 1.5 !important;
  color: var(--text-secondary) !important;
  margin: 0 !important;
}

/* 11. Dataset Explorer */
[data-testid="stMetric"] {
  background-color: var(--bg-card);
  border: 1px solid var(--border-color);
  border-radius: 8px;
  padding: 12px 14px;
}
[data-testid="stMetricLabel"] {
  font-size: 11px !important;
  text-transform: uppercase !important;
  letter-spacing: 0.04em !important;
  font-weight: 600 !important;
  color: var(--text-secondary) !important;
}
[data-testid="stMetricValue"] {
  font-family: var(--font-serif) !important;
  font-size: 24px !important;
  font-weight: 600 !important;
  color: var(--text-primary) !important;
}

.domain-grid-4x2 {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 12px;
  margin: 16px 0 24px;
}
.domain-plain-card {
  background-color: var(--bg-card);
  border: 1px solid var(--border-color);
  border-radius: 8px;
  padding: 14px;
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.domain-icon-wrap {
  width: 24px;
  height: 24px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  color: var(--accent);
}
.domain-title-sans {
  font-size: 14px;
  font-weight: 600;
  color: var(--text-primary);
}
.domain-mono-count {
  font-family: var(--font-mono);
  font-size: 12px;
  color: var(--text-secondary);
}

[data-testid="stDataFrame"] {
  border: 1px solid var(--border-color) !important;
  border-radius: 8px !important;
  background-color: var(--bg-card) !important;
}
[data-testid="stDataFrame"] div[data-testid="glide-cell"] {
  background-color: var(--bg-card) !important;
  color: var(--text-primary) !important;
  border-bottom: 1px solid var(--border-color) !important;
  font-size: 14px !important;
}
[data-testid="stDataFrame"] th,
[data-testid="stDataFrame"] [role="columnheader"] {
  background-color: var(--bg-card-subtle) !important;
  color: var(--text-primary) !important;
  font-weight: 600 !important;
  border-bottom: 1px solid var(--border-color) !important;
}

[data-testid="stSelectbox"] > div {
  background-color: var(--bg-card) !important;
  border: 1px solid var(--border-color) !important;
  border-radius: 8px !important;
}

/* 12. Algorithm Architecture */
.algo-section {
  padding: 24px 0;
}
.algo-divider {
  border: none;
  border-top: 1px solid var(--border-color);
  margin: 0;
}
.algo-header-row {
  display: flex;
  align-items: baseline;
  gap: 16px;
  margin-bottom: 14px;
}
.algo-num-serif {
  font-family: var(--font-serif);
  font-size: 28px;
  font-weight: 600;
  color: var(--accent);
  line-height: 1;
}
.algo-title-serif {
  font-family: var(--font-serif) !important;
  font-size: 20px !important;
  font-weight: 600 !important;
  color: var(--text-primary) !important;
  margin: 0 !important;
}
.algo-content-grid {
  display: grid;
  grid-template-columns: 1fr 1.6fr;
  gap: 28px;
}
.algo-eyebrow {
  font-size: 12px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  color: var(--text-secondary);
  margin-bottom: 6px;
}
.algo-paragraph {
  font-size: 15px;
  line-height: 1.6;
  color: var(--text-primary);
  margin: 0;
}

.katex-display {
  font-size: 15px !important;
  margin: 12px 0 !important;
}
.katex {
  font-size: 15px !important;
  color: var(--text-primary) !important;
}

/* 13. Footer */
.app-footer {
  display: flex;
  justify-content: space-between;
  align-items: center;
  border-top: 1px solid var(--border-color);
  margin-top: 48px;
  padding-top: 16px;
  padding-bottom: 24px;
  font-size: 13px;
  color: var(--text-secondary);
}

@media (max-width: 900px) {
  .plain-header { flex-direction: column; align-items: flex-start; }
  .kpi-row-grid { grid-template-columns: repeat(2, 1fr); }
  .domain-grid-4x2 { grid-template-columns: repeat(2, 1fr); }
  .algo-content-grid { grid-template-columns: 1fr; gap: 16px; }
}
"""


def get_global_styles() -> str:
    """Returns pure <style> tag with zero leading indentation for st.markdown / st.html."""
    return f"<style>\n{_CSS_BODY.strip()}\n</style>"
