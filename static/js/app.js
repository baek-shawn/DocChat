// DocChat 프론트엔드 — 프레임워크 없는 순수 JS. 하나의 `state` 객체와 `render()`로 화면을 그린다.
// 모델 출력은 절대 innerHTML로 넣지 않는다(항상 textContent / createElement).
(() => {
  'use strict';

  const MAX_FILES = 12;
  const MAX_FILE_BYTES = 64 * 1024 * 1024;
  const LOCAL_PROVIDER = 'openaiCompatible';
  const PROVIDER_LABELS = { openaiCompatible: '로컬 API', openai: 'OpenAI', anthropic: 'Anthropic', gemini: 'Gemini' };
  const TYPE_LABELS = { text: '텍스트', object: '객체', table: '표', dimension: '치수', stamp: '도장', signature: '서명', diagram: '다이어그램', other: '기타' };
  const IMAGE_MODE_LABELS = { whole: '전체', tile: '타일' };
  // 답변(추론) 호출에 싣는 이미지(Step 8) — app/config.py의 ANSWER_IMAGE_MODES와 짝이다.
  const ANSWER_IMAGE_LABELS = { off: '끔', uploads: '업로드 이미지만', whole: '전체' };
  // 턴 트레이스(Step 7)의 이벤트 종류·상태 — app/trace.py의 KINDS·STATUSES와 짝이다.
  const TRACE_KINDS = { input: '입력', preprocess: '전처리', ocr: '전사', evidence: '증거', model: '모델', tool: '도구', loop: '루프', cleanup: '정리', progress: '진행', files: '파일', answer: '답변' };
  const TRACE_STATUS = { running: '진행 중', done: '완료', failed: '실패', cancelled: '취소', interrupted: '중단됨' };
  const MODEL_KIND_LABELS = { answer: '답변', ocr: '전사', grounding: '위치 확인' };

  const store = {
    get(key, fallback = '') { try { return localStorage.getItem(`docchat.${key}`) ?? fallback; } catch { return fallback; } },
    set(key, value) { try { localStorage.setItem(`docchat.${key}`, String(value)); } catch { /* 저장소 사용 불가 */ } },
    // API key는 탭을 닫으면 사라지는 sessionStorage에만 둔다.
    getKey(provider) { try { return sessionStorage.getItem(`docchat.apiKey.${provider}`) || ''; } catch { return ''; } },
    setKey(provider, value) { try { sessionStorage.setItem(`docchat.apiKey.${provider}`, value); } catch { /* 무시 */ } }
  };

  const state = {
    sessions: [],
    currentId: '',
    messages: [],          // {role, content, files?, artifacts?, pending?, activity?, error?}
    pendingFiles: [],      // 아직 보내지 않은 첨부 {name, mime, size, base64, status, tone}
    busy: false,
    abort: null,
    editingIndex: -1,
    selecting: false,
    selected: new Set(),
    provider: store.get('provider', LOCAL_PROVIDER),
    baseUrl: store.get('baseUrl', 'http://127.0.0.1:11434/v1'),
    model: store.get('model', ''),
    contextSize: Number(store.get('contextSize', '8192')) || 8192,
    disableThinking: store.get('disableThinking', 'true') !== 'false',
    imageMode: store.get('imageMode', ''),   // '' = 고른 적 없음 → 서버 기본값(DOCCHAT_IMAGE_MODE)을 따른다
    serverImageMode: 'whole',
    tiling: null,                            // 서버에 적용 중인 타일 설정(/api/health)
    answerImageMode: store.get('answerImageMode', ''),   // '' = 고른 적 없음 → 서버 기본값(DOCCHAT_ANSWER_IMAGE_MODE)
    serverAnswerImageMode: 'uploads',
    maxModelImages: 12,                      // 답변 호출 한 번에 싣는 이미지 수 상한(/api/health)
    // 호출 종류별 추론 끄기. 'true' | 'false' | ''(고른 적 없음 → 서버 기본값)
    disableThinkingGrounding: store.get('disableThinkingGrounding', ''),
    disableThinkingOcr: store.get('disableThinkingOcr', ''),
    serverVision: { disableThinkingGrounding: true, disableThinkingOcr: true, maxTokens: 4096 },
    serverReasoning: null,                   // 추론 제어(Step 6) 설정: 호출 종류별 추론 예산·반복 기준(/api/health)
    serverDebugTrace: false,                 // 서버가 턴 트레이스를 기록하는지(/api/health)
    trace: { id: '', doc: null, timer: 0, open: new Set(), sections: new Map(), collapsed: new Set() },   // 열려 있는 "턴 과정" 창(펼친 행·접이식·접은 가지 기억)
    models: []
  };
  if (!PROVIDER_LABELS[state.provider]) state.provider = LOCAL_PROVIDER;
  if (!IMAGE_MODE_LABELS[state.imageMode]) state.imageMode = '';
  if (!ANSWER_IMAGE_LABELS[state.answerImageMode]) state.answerImageMode = '';

  const $ = (id) => document.getElementById(id);
  const els = Object.fromEntries([
    'shell', 'newChat', 'selectSessions', 'refreshSessions', 'sessionList', 'deleteSelected', 'deleteAll',
    'modelSelect', 'testConnection', 'providerBadge', 'themeToggle', 'openSettings', 'messages', 'pendingFiles',
    'dropZone', 'attach', 'prompt', 'send', 'fileInput', 'viewer', 'viewerResize', 'viewerTitle', 'viewerMeta',
    'toggleLabels', 'zoomOut', 'zoomReset', 'zoomIn', 'viewerDownload', 'viewerClose', 'viewerStage', 'viewerRegions',
    'settingsDialog', 'providerSelect', 'baseUrlField', 'baseUrl', 'apiKey', 'apiKeyLabel', 'modelName', 'modelOptions',
    'contextField', 'contextSize', 'disableThinking', 'callThinking', 'disableThinkingGrounding', 'disableThinkingOcr', 'callThinkingHelp',
    'imageMode', 'imageModeHelp', 'answerImageMode', 'answerImageHelp', 'traceHelp', 'settingsTest', 'testDot', 'testText', 'saveSettings',
    'traceDialog', 'traceStatus', 'traceRefresh', 'traceFoldAll', 'traceUnfoldAll', 'traceDownload', 'traceClose', 'traceSummary', 'traceBody'
  ].map((id) => [id, $(id)]));

  const el = (tag, className, text) => {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  };
  const formatSize = (size) => size < 1024 ? `${size} B` : size < 1048576 ? `${(size / 1024).toFixed(1)} KB` : `${(size / 1048576).toFixed(1)} MB`;
  const attachmentUrl = (id, download) => `/api/attachments/${id}/content${download ? '?download=true' : ''}`;
  const connection = () => ({ provider: state.provider, apiKey: store.getKey(state.provider), baseUrl: state.provider === LOCAL_PROVIDER ? state.baseUrl : '' });

  // ------------------------------------------------------------------ HTTP
  async function request(path, { method = 'GET', body, signal } = {}) {
    const init = { method, signal, headers: {} };
    if (body !== undefined) { init.headers['Content-Type'] = 'application/json'; init.body = JSON.stringify(body); }
    const response = await fetch(path, init);
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.error || `요청 실패: HTTP ${response.status}`);
    return data;
  }

  // /api/chat은 NDJSON(한 줄에 JSON 하나)으로 진행 상황을 흘려보낸다. SSE가 아니다.
  async function streamChat(body, signal, onEvent) {
    const response = await fetch('/api/chat', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body), signal });
    if (!response.ok || !response.body) {
      const data = await response.json().catch(() => ({}));
      throw new Error(data.error || `요청 실패: HTTP ${response.status}`);
    }
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let final = null;
    const handle = (line) => {
      if (!line.trim()) return;
      let event;
      try { event = JSON.parse(line); } catch { return; }
      if (event.type === 'error') throw new Error(event.error || '알 수 없는 오류');
      if (event.type === 'final') final = event;
      onEvent(event);
    };
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';
      lines.forEach(handle);
    }
    handle(buffer);
    if (!final) throw new Error('서버가 응답을 끝내기 전에 연결이 닫혔습니다.');
    return final;
  }

  // ------------------------------------------------------------------ 세션
  async function loadSessions() {
    state.sessions = (await request('/api/sessions')).sessions || [];
    state.selected = new Set([...state.selected].filter((id) => state.sessions.some((item) => item.id === id)));
    renderSessions();
  }

  function newChat() {
    if (state.busy) return;
    Object.assign(state, { currentId: '', messages: [], pendingFiles: [], editingIndex: -1 });
    els.prompt.value = '';
    closeViewer();
    autoGrow(); render(); renderSessions();
    els.prompt.focus();
  }

  async function openSession(id) {
    if (state.busy || id === state.currentId) return;
    const session = await request(`/api/sessions/${encodeURIComponent(id)}`);
    Object.assign(state, { currentId: session.id, messages: session.messages || [], pendingFiles: [], editingIndex: -1 });
    closeViewer();
    await attachOrphanTraces(session.id);
    render(); renderSessions();
  }

  // 답변이 저장되지 않은 턴(중지·오류·연결 끊김)의 트레이스는 답변 메시지가 없어 버튼을 걸 곳이 없다
  // → 그 질문 메시지 아래에 단다. 트레이스가 꺼진 서버는 빈 목록을 돌려준다.
  async function attachOrphanTraces(sessionId) {
    let traces = [];
    try { traces = (await request(`/api/sessions/${encodeURIComponent(sessionId)}/traces`)).traces || []; } catch { return; }
    const known = new Set(state.messages.map((message) => message.meta?.traceId).filter(Boolean));
    for (const item of traces) {
      if (known.has(item.id)) continue;
      const target = [...state.messages].reverse().find((message) => message.role === 'user' && (message.createdAt || 0) <= item.createdAt + 5000);
      if (target && !target.traceId) { target.traceId = item.id; target.traceStatus = item.status; }
    }
  }

  async function deleteSession(id) {
    if (state.busy || !confirm('이 세션을 삭제할까요? 첨부와 대화가 모두 지워집니다.')) return;
    await request(`/api/sessions/${encodeURIComponent(id)}`, { method: 'DELETE' });
    if (state.currentId === id) newChat();
    await loadSessions();
  }

  function toggleSelecting() {
    state.selecting = !state.selecting;
    if (!state.selecting) state.selected.clear();
    els.selectSessions.textContent = state.selecting ? '취소' : '선택';
    els.deleteSelected.hidden = !state.selecting;
    renderSessions();
  }

  async function deleteSelected() {
    const ids = [...state.selected];
    if (!ids.length || !confirm(`선택한 세션 ${ids.length}개를 삭제할까요?`)) return;
    await request('/api/sessions', { method: 'DELETE', body: { ids } });
    if (ids.includes(state.currentId)) newChat();
    toggleSelecting();
    await loadSessions();
  }

  async function deleteAll() {
    if (!state.sessions.length || !confirm(`세션 ${state.sessions.length}개를 모두 삭제할까요? 되돌릴 수 없습니다.`)) return;
    await request('/api/sessions', { method: 'DELETE', body: { all: true } });
    newChat();
    await loadSessions();
  }

  function renderSessions() {
    els.sessionList.replaceChildren();
    if (!state.sessions.length) { els.sessionList.append(el('div', 'empty-hint', '대화를 시작하면 여기에 세션이 쌓입니다.')); return; }
    for (const session of state.sessions) {
      const row = el('div', `session${session.id === state.currentId ? ' active' : ''}${state.selecting ? ' selecting' : ''}`);
      if (state.selecting) {
        const check = el('input', 'session-check');
        check.type = 'checkbox'; check.checked = state.selected.has(session.id);
        check.addEventListener('change', () => { check.checked ? state.selected.add(session.id) : state.selected.delete(session.id); });
        row.append(check);
      }
      const open = el('button', 'session-open');
      open.title = session.title;
      open.append(el('span', 'session-title', session.title || '새 채팅'));
      const files = session.fileCount ? ` · 첨부 ${session.fileCount}` : '';
      open.append(el('span', 'session-meta', `${new Date(session.updatedAt).toLocaleString('ko-KR', { dateStyle: 'short', timeStyle: 'short' })}${files}`));
      open.addEventListener('click', () => openSession(session.id).catch((error) => alert(error.message)));
      row.append(open);
      if (!state.selecting) {
        const remove = el('button', 'session-delete', '×');
        remove.title = '세션 삭제';
        remove.addEventListener('click', () => deleteSession(session.id).catch((error) => alert(error.message)));
        row.append(remove);
      }
      els.sessionList.append(row);
    }
  }

  // ------------------------------------------------------------------ 모델 설정
  function persistSettings() {
    store.set('provider', state.provider); store.set('baseUrl', state.baseUrl);
    store.set('model', state.model); store.set('contextSize', state.contextSize); store.set('disableThinking', state.disableThinking);
    store.set('imageMode', state.imageMode); store.set('answerImageMode', state.answerImageMode);
    store.set('disableThinkingGrounding', state.disableThinkingGrounding); store.set('disableThinkingOcr', state.disableThinkingOcr);
  }

  // 요청에 실을 이미지 처리 방식. 사용자가 고른 적이 없으면 서버 기본값.
  const imageMode = () => state.imageMode || state.serverImageMode;
  // 요청에 실을 답변 호출 이미지 모드(Step 8). 고른 적이 없으면 서버 기본값.
  const answerImageMode = () => state.answerImageMode || state.serverAnswerImageMode;
  // 요청에 실을 호출별 추론 끄기(kind: 'disableThinkingGrounding' | 'disableThinkingOcr'). 고른 적이 없으면 서버 기본값.
  const callThinkingOff = (kind) => state[kind] === '' ? state.serverVision[kind] !== false : state[kind] === 'true';

  async function loadServerDefaults() {
    try {
      const health = await request('/api/health');
      if (IMAGE_MODE_LABELS[health.imageMode]) state.serverImageMode = health.imageMode;
      state.tiling = health.tiling || null;
      if (ANSWER_IMAGE_LABELS[health.answerImageMode]) state.serverAnswerImageMode = health.answerImageMode;
      if (Number(health.maxModelImages) > 0) state.maxModelImages = Number(health.maxModelImages);
      if (health.vision) state.serverVision = { ...state.serverVision, ...health.vision };
      state.serverReasoning = health.reasoning || null;
      state.serverDebugTrace = health.debugTrace === true;
    } catch { /* 기본값(전체)으로 둔다 */ }
  }

  // "추론 끄기"(모든 호출)가 켜져 있으면 호출별 선택은 적용되지 않는다 → 고른 값은 그대로 두고 잠가서 보여 준다.
  function syncCallThinking() {
    const everything = els.disableThinking.checked;
    els.disableThinkingGrounding.disabled = everything; els.disableThinkingOcr.disabled = everything;
    els.callThinking.classList.toggle('inactive', everything);
    const limit = state.serverVision.maxTokens;
    const budget = state.serverReasoning?.budget;
    const fmt = (value) => Number(value || 0).toLocaleString();
    els.callThinkingHelp.textContent = (everything
      ? '위 항목이 켜져 있어 모든 호출의 추론이 꺼집니다. 위 항목을 끄면 아래 두 선택이 적용됩니다.'
      : '답변 호출만 추론을 쓰고, 위치 확인과 전사는 끌 수 있습니다. 이 둘은 보이는 것을 옮겨 적는 호출이라, 추론을 켜면 같은 생각을 맴돌다 끝나지 않는 일이 있습니다.')
      + (budget
        ? ` 추론을 켠 호출은 추론 예산(답변 ${fmt(budget.answer)} · 위치 확인 ${fmt(budget.grounding)} · 전사 ${fmt(budget.ocr)}토큰)을 넘거나 같은 내용을 되풀이하면 추론을 끊고 답만 이어 쓰게 합니다(.env의 DOCCHAT_REASONING_*).`
        : '')
      + (limit > 0
        ? ` 위치 확인·전사 호출의 출력 상한은 ${fmt(limit)}토큰(.env의 DOCCHAT_VISION_MAX_TOKENS)이고, 추론을 켠 호출에는 추론 예산이 더해집니다. 상한에 닿은 호출은 다시 보내지 않습니다.`
        : ' 위치 확인·전사 호출의 출력 상한이 꺼져 있습니다(.env의 DOCCHAT_VISION_MAX_TOKENS=0).');
  }

  function renderModelBar() {
    const options = [...new Set([state.model, ...state.models].filter(Boolean))];
    els.modelSelect.replaceChildren(new Option(options.length ? '모델 선택' : '설정에서 모델을 지정하세요', ''));
    options.forEach((id) => els.modelSelect.append(new Option(id, id)));
    els.modelSelect.value = state.model;
    els.providerBadge.textContent = state.provider === LOCAL_PROVIDER ? `${PROVIDER_LABELS[state.provider]} · ${state.baseUrl.replace(/^https?:\/\//, '')}` : PROVIDER_LABELS[state.provider];
  }

  async function loadModels({ silent = true } = {}) {
    try {
      state.models = (await request('/api/models', { method: 'POST', body: connection() })).models || [];
      if (!state.model && state.models.length) { state.model = state.models[0]; persistSettings(); }
    } catch (error) {
      state.models = [];
      if (!silent) alert(error.message);
    }
    renderModelBar();
  }

  function syncSettingsForm() {
    const provider = els.providerSelect.value;
    const local = provider === LOCAL_PROVIDER;
    els.baseUrlField.hidden = !local;
    els.contextField.hidden = !local;
    els.apiKeyLabel.textContent = local ? 'API key (인증이 없는 로컬 서버는 비워 두세요)' : 'API key';
    els.apiKey.value = store.getKey(provider);
    els.modelName.value = provider === state.provider ? state.model : '';
    els.modelOptions.replaceChildren();
    if (provider === state.provider) state.models.forEach((id) => els.modelOptions.append(new Option(id, id)));
    setTestResult('idle', '테스트 전');
  }

  function openSettings() {
    els.providerSelect.value = state.provider;
    els.baseUrl.value = state.baseUrl;
    els.contextSize.value = state.contextSize;
    els.disableThinking.checked = state.disableThinking;
    els.disableThinkingGrounding.checked = callThinkingOff('disableThinkingGrounding');
    els.disableThinkingOcr.checked = callThinkingOff('disableThinkingOcr');
    syncCallThinking();
    els.imageMode.value = imageMode();
    const tiling = state.tiling;
    els.imageModeHelp.textContent = '전사(OCR)와 위치 확인(bbox) 호출에 적용됩니다. 타일은 작은 글자·심볼을 더 크게 보여 주지만 호출 수가 늘어 느려집니다. 요청마다 적용되므로 같은 문서를 두 방식으로 비교할 수 있습니다.'
      + (tiling ? ` 현재 타일 설정: ${tiling.tileSize}px · 겹침 ${Math.round(tiling.overlap * 1000) / 10}% · ${tiling.renderDpi}DPI · 긴 변 ${tiling.minSourceEdge}px 이하는 나누지 않음 · 장당 최대 ${tiling.maxTiles}타일.` : '');
    els.answerImageMode.value = answerImageMode();
    els.answerImageHelp.textContent = '답변(추론) 호출에 어떤 이미지를 실을지입니다. "끔"은 텍스트(네이티브·전사)와 위치 확인 도구만 씁니다. "업로드 이미지만"은 지금까지의 동작입니다(PDF 쪽은 텍스트만). '
      + '"전체"는 PDF의 모든 쪽(글자가 충분한 쪽도)을 한 장씩 실어 그림에만 있는 것(형상·심볼·배치)을 볼 수 있게 하지만, 도구 루프의 호출마다 이미지 토큰이 들고 컨텍스트가 작은 로컬 모델은 넘칠 수 있습니다. '
      + `한 번에 최대 ${state.maxModelImages}장(.env의 DOCCHAT_MAX_MODEL_IMAGES), 넘치면 쪽 순서로 앞에서부터. 요청마다 적용되므로 같은 질문을 방식별로 비교할 수 있습니다.`;
    els.traceHelp.textContent = state.serverDebugTrace
      ? '턴 과정 기록(개발용)이 켜져 있습니다. 답변마다 "과정 보기"로 전처리·전사·모델 호출·도구 실행을 시간순으로 볼 수 있습니다(.env의 DOCCHAT_DEBUG_TRACE).'
      : '턴 과정 기록(개발용)은 꺼져 있습니다. 켜려면 .env에 DOCCHAT_DEBUG_TRACE=1을 적고 서버를 다시 띄우세요.';
    syncSettingsForm();
    if (!els.settingsDialog.open) els.settingsDialog.showModal();
  }

  function setTestResult(status, text) {
    els.testDot.dataset.status = status;
    els.testText.textContent = text; els.testText.title = text;
  }

  function formConnection() {
    const provider = els.providerSelect.value;
    return { provider, apiKey: els.apiKey.value.trim(), baseUrl: provider === LOCAL_PROVIDER ? els.baseUrl.value.trim().replace(/\/+$/, '') : '' };
  }

  async function testFromSettings() {
    els.settingsTest.disabled = true; setTestResult('busy', '확인 중…');
    try {
      const data = await request('/api/test-connection', { method: 'POST', body: { ...formConnection(), model: els.modelName.value.trim() } });
      setTestResult(data.ok ? 'ok' : 'fail', data.message);
      if (data.ok) {
        els.modelOptions.replaceChildren();
        (data.models || []).forEach((id) => els.modelOptions.append(new Option(id, id)));
        if (!els.modelName.value.trim() && data.models?.length) els.modelName.value = data.models[0];
      }
    } catch (error) { setTestResult('fail', error.message); } finally { els.settingsTest.disabled = false; }
  }

  function saveSettings() {
    const form = formConnection();
    if (form.provider === LOCAL_PROVIDER && !/^https?:\/\/.+/i.test(form.baseUrl)) { setTestResult('fail', 'http:// 또는 https://로 시작하는 서버 주소를 입력하세요.'); els.baseUrl.focus(); return; }
    if (form.provider !== LOCAL_PROVIDER && !form.apiKey) { setTestResult('fail', 'API key를 입력하세요.'); els.apiKey.focus(); return; }
    state.provider = form.provider;
    if (form.provider === LOCAL_PROVIDER) state.baseUrl = form.baseUrl;
    state.model = els.modelName.value.trim();
    state.contextSize = Math.max(1024, Number(els.contextSize.value) || 8192);
    state.disableThinking = els.disableThinking.checked;
    state.disableThinkingGrounding = String(els.disableThinkingGrounding.checked);
    state.disableThinkingOcr = String(els.disableThinkingOcr.checked);
    state.imageMode = IMAGE_MODE_LABELS[els.imageMode.value] ? els.imageMode.value : '';
    state.answerImageMode = ANSWER_IMAGE_LABELS[els.answerImageMode.value] ? els.answerImageMode.value : '';
    store.setKey(form.provider, form.apiKey);
    persistSettings();
    els.settingsDialog.close();
    renderModelBar();
    void loadModels();
  }

  async function testConnection() {
    els.testConnection.disabled = true;
    const previous = els.testConnection.textContent; els.testConnection.textContent = '확인 중…';
    try { alert((await request('/api/test-connection', { method: 'POST', body: { ...connection(), model: state.model } })).message); }
    catch (error) { alert(error.message); }
    finally { els.testConnection.disabled = false; els.testConnection.textContent = previous; }
  }

  // ------------------------------------------------------------------ 첨부
  const toBase64 = (file) => new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result).split(',')[1] || '');
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(file);
  });

  function mimeOf(file) {
    if (file.type) return file.type === 'image/jpg' ? 'image/jpeg' : file.type;
    const extension = (file.name.split('.').pop() || '').toLowerCase();
    return { pdf: 'application/pdf', png: 'image/png', jpg: 'image/jpeg', jpeg: 'image/jpeg', webp: 'image/webp', gif: 'image/gif', bmp: 'image/bmp', tif: 'image/tiff', tiff: 'image/tiff' }[extension] || '';
  }

  async function addFiles(files) {
    for (const file of files) {
      if (state.pendingFiles.length >= MAX_FILES) { alert(`한 번에 최대 ${MAX_FILES}개까지 첨부할 수 있습니다.`); break; }
      const mime = mimeOf(file);
      if (mime !== 'application/pdf' && !mime.startsWith('image/')) { alert(`${file.name}: PDF와 이미지만 첨부할 수 있습니다.`); continue; }
      if (file.size > MAX_FILE_BYTES) { alert(`${file.name}: 64MB를 넘는 파일은 첨부할 수 없습니다.`); continue; }
      const item = { name: file.name, mime, size: file.size, base64: '', status: '읽는 중…', tone: '' };
      state.pendingFiles.push(item); renderPendingFiles();
      try {
        item.base64 = await toBase64(file);
        item.status = '검사 중…'; renderPendingFiles();
        // 업로드 직후 서버에 미리 검사시켜 어떤 경로로 읽힐지 보여 준다.
        const report = await request('/api/attachments/inspect', { method: 'POST', body: { attachment: { name: item.name, mime, size: item.size, base64: item.base64 } } });
        if (report.kind === 'pdf') {
          const pages = `${report.totalPages}쪽`;
          item.status = report.visualPages > 0 ? `${pages} · ${report.visualPages}쪽은 비전 전사 필요` : `${pages} · 네이티브 텍스트 ${report.parsedCharacters.toLocaleString()}자`;
          item.tone = report.visualPages > 0 ? 'warn' : '';
          if (report.truncated) item.status += ` (앞 ${report.processedPages}쪽만 검사)`;
          // 답변 호출 이미지 "전체"(Step 8)면 글자가 충분한 쪽도 이미지로 실린다.
          if (answerImageMode() === 'whole') item.status += ` · 답변 호출에 쪽 이미지 포함(최대 ${state.maxModelImages}장)`;
        } else {
          item.status = report.resized ? `${report.sourceWidth}×${report.sourceHeight} → ${report.width}×${report.height}로 축소해 전달` : `${report.width}×${report.height} 원본 그대로 전달`;
          if (answerImageMode() === 'off') item.status += ' · 답변 호출에는 싣지 않음(끔)';
          // 타일 모드에서도 답변 호출에는 위의 한 장이 간다. 위치 확인(bbox)만 원본을 타일로 나눠 본다.
          const longEdge = Math.max(report.sourceWidth || 0, report.sourceHeight || 0);
          if (imageMode() === 'tile' && state.tiling && longEdge > Math.max(state.tiling.minSourceEdge, state.tiling.tileSize)) item.status += ' · 위치 확인은 원본을 타일로';
        }
      } catch (error) { item.status = `오류: ${error.message}`; item.tone = 'bad'; item.failed = true; }
      renderPendingFiles();
    }
  }

  function renderPendingFiles() {
    els.pendingFiles.replaceChildren();
    state.pendingFiles.forEach((file, index) => {
      const chip = el('div', `chip ${file.tone || ''}`);
      chip.append(el('span', '', `${file.name} · ${formatSize(file.size)}`), el('span', 'status', file.status));
      const remove = el('button', 'x', '×'); remove.title = '첨부 취소';
      remove.addEventListener('click', () => { state.pendingFiles.splice(index, 1); renderPendingFiles(); });
      chip.append(remove);
      els.pendingFiles.append(chip);
    });
  }

  // ------------------------------------------------------------------ 전송
  async function send(overrideText) {
    const text = (typeof overrideText === 'string' ? overrideText : els.prompt.value).trim();
    const files = state.pendingFiles.filter((file) => !file.failed && file.base64);
    if ((!text && !files.length) || state.busy) return;
    if (!state.model || (state.provider !== LOCAL_PROVIDER && !store.getKey(state.provider))) { openSettings(); return; }

    if (state.editingIndex >= 0) state.messages.splice(state.editingIndex);   // 수정·재전송: 그 지점부터 다시 쓴다
    state.editingIndex = -1;
    const userMessage = { role: 'user', content: text || '첨부한 파일을 분석해 주세요.', createdAt: Date.now() };
    if (files.length) userMessage.files = files.map((file) => ({ name: file.name, kind: file.mime === 'application/pdf' ? 'pdf' : 'image', size: file.size }));
    const history = [...state.messages.filter((message) => !message.pending), userMessage];
    const placeholder = { role: 'assistant', content: '', pending: true, activity: ['요청을 보내는 중…'], artifacts: [] };
    state.messages = [...history, placeholder];
    state.pendingFiles = [];
    els.prompt.value = ''; autoGrow();
    state.busy = true; state.abort = new AbortController();
    render();

    try {
      const final = await streamChat({
        ...connection(), model: state.model, contextSize: state.contextSize, disableThinking: state.disableThinking,
        disableThinkingGrounding: callThinkingOff('disableThinkingGrounding'), disableThinkingOcr: callThinkingOff('disableThinkingOcr'),
        imageMode: imageMode(), answerImageMode: answerImageMode(), conversationId: state.currentId, stream: true,
        // meta(답변을 어떤 방식으로 처리했는지)도 되돌려 보내야 서버가 대화를 다시 저장할 때 지워지지 않는다.
        messages: history.map(({ role, content, files: sent, artifacts, meta, createdAt }) => ({ role, content, files: sent, artifacts, meta, createdAt })),
        attachments: files.map(({ name, mime, size, base64 }) => ({ name, mime, size, base64 }))
      }, state.abort.signal, (event) => {
        if (event.type === 'conversation') {
          state.currentId = event.conversationId;
          // 트레이스를 켠 서버는 id를 먼저 알려 준다 → 답이 나오기 전에도 "과정 보기"를 열 수 있다.
          if (event.traceId) { placeholder.traceId = event.traceId; render(); }
        }
        if (event.type === 'progress') {
          if (event.live) {
            // "추론 중… n토큰"처럼 1초마다 오는 문구(Step 6)는 줄을 늘리지 않고 마지막 live 줄을 바꿔 쓴다.
            if (placeholder.liveIndex === placeholder.activity.length - 1 && placeholder.liveIndex >= 0) placeholder.activity[placeholder.liveIndex] = event.message;
            else { placeholder.activity.push(event.message); placeholder.liveIndex = placeholder.activity.length - 1; }
            render();
          } else if (placeholder.activity.at(-1) !== event.message) { placeholder.activity.push(event.message); placeholder.liveIndex = -1; render(); }
        }
      });
      Object.assign(placeholder, { content: final.text, artifacts: final.artifacts || [], meta: final.meta || {}, pending: false });
      if (final.files?.length) userMessage.files = final.files;               // 저장된 첨부 id가 붙어 돌아온다
      const withBoxes = [...placeholder.artifacts].reverse().find((artifact) => artifact.view === 'image');
      if (withBoxes) openArtifact(withBoxes);
    } catch (error) {
      const stopped = error.name === 'AbortError';
      Object.assign(placeholder, { pending: false, error: !stopped, content: stopped ? '생성을 중지했습니다. 메시지를 수정하거나 다시 보낼 수 있습니다.' : `오류: ${error.message}` });
    } finally {
      state.busy = false; state.abort = null;
      render();
      loadSessions().catch(() => {});
    }
  }

  function editMessage(index) {
    if (state.busy) return;
    state.editingIndex = index;
    els.prompt.value = state.messages[index].content;
    autoGrow(); els.prompt.focus(); render();
  }

  function resendMessage(index) {
    if (state.busy) return;
    state.editingIndex = index;
    void send(state.messages[index].content);
  }

  // ------------------------------------------------------------------ 렌더링
  function render() {
    els.messages.replaceChildren();
    if (!state.messages.length) {
      const welcome = el('div', 'welcome');
      welcome.append(el('h1', '', '문서를 올리고 물어보세요'), el('div', '', 'PDF와 이미지를 로컬 또는 클라우드 비전 모델로 분석합니다.'));
      const list = el('ul');
      ['“이 도면의 도면 번호와 개정 이력을 표로 정리해 줘”', '“부품표(BOM)를 추출해 줘”', '“승인 도장과 서명 위치를 표시해 줘” → 이미지 위에 영역이 표시됩니다'].forEach((tip) => list.append(el('li', '', tip)));
      welcome.append(list);
      els.messages.append(welcome);
    }
    state.messages.forEach((message, index) => els.messages.append(renderMessage(message, index)));
    els.send.textContent = state.busy ? '■' : '↑';
    els.send.title = state.busy ? '생성 중지' : '보내기';
    els.prompt.placeholder = state.editingIndex >= 0 ? '메시지를 수정한 뒤 보내세요' : '문서에 대해 물어보세요. PDF·이미지를 끌어다 놓을 수 있습니다.';
    renderPendingFiles();
    requestAnimationFrame(() => { els.messages.scrollTop = els.messages.scrollHeight; });
  }

  function renderMessage(message, index) {
    const isBot = message.role === 'assistant';
    const wrap = el('article', `msg ${message.role}${message.pending ? ' pending' : ''}${message.error || /^오류:/.test(message.content) && isBot ? ' error' : ''}`);
    wrap.append(el('div', `avatar${isBot ? ' bot' : ''}`, isBot ? 'AI' : '나'));
    const body = el('div', 'msg-body');
    body.append(el('div', 'msg-name', isBot ? 'DocChat' : '나'));

    if (message.pending) {
      const log = el('div', 'activity');
      (message.activity || []).forEach((text, step, all) => {
        const active = step === all.length - 1;
        const row = el('div', `step${active ? '' : ' done'}`);
        row.append(active ? el('span', 'spinner') : el('span', 'tick', '✓'), el('span', '', text));
        log.append(row);
      });
      body.append(log);
    } else if (isBot) {
      const content = el('div'); renderMarkdown(content, message.content); body.append(content);
    } else {
      body.append(el('div', 'msg-text', message.content));
    }

    if (message.files?.length) {
      const row = el('div', 'file-row');
      for (const file of message.files) {
        const openable = Number.isInteger(file.attachmentId);
        const chip = el(openable ? 'button' : 'span', 'chip');
        chip.append(el('span', '', `📎 ${file.name} · ${formatSize(file.size || 0)}`));
        if (openable) {
          chip.title = file.kind === 'pdf' ? '새 탭에서 PDF 열기' : '뷰어에서 이미지 열기';
          chip.addEventListener('click', () => file.kind === 'pdf' ? window.open(attachmentUrl(file.attachmentId), '_blank', 'noopener') : openArtifact({ name: file.name, title: file.name, mime: file.mime || 'image/png', view: 'image', attachmentId: file.attachmentId }));
        }
        row.append(chip);
      }
      body.append(row);
    }

    const viewable = (message.artifacts || []).filter((artifact) => artifact.view === 'image' && Number.isInteger(artifact.attachmentId));
    if (viewable.length) {
      const row = el('div', 'artifact-row');
      for (const artifact of viewable) {
        const count = artifact.boxes?.length || 0;
        const button = el('button', 'artifact-btn', `🔍 ${artifact.name} · ${count ? `영역 ${count}개 보기` : '이미지 보기'}`);
        button.addEventListener('click', () => openArtifact(artifact));
        row.append(button);
      }
      body.append(row);
    }

    const processing = isBot && !message.pending ? describeProcessing(message.meta) : '';
    if (processing) body.append(el('div', 'msg-meta', processing));

    // 턴 트레이스(개발용): 이 답을 만든 과정. 진행 중인 턴도 열리고, 답변이 저장되지 않은 턴은 질문 아래에 붙는다.
    const traceId = message.meta?.traceId || message.traceId || '';
    if (traceId) {
      const orphan = { running: '끝나지 않은 턴', failed: '실패한 턴', cancelled: '취소된 턴', interrupted: '중단된 턴' };
      const suffix = message.pending ? ' (진행 중)' : (message.traceStatus && message.traceStatus !== 'done' ? ` (${orphan[message.traceStatus] || message.traceStatus})` : '');
      const open = el('button', 'trace-btn', `⏱ 과정 보기${suffix}`);
      open.addEventListener('click', () => openTrace(traceId));
      const row = el('div', 'trace-row'); row.append(open); body.append(row);
    }

    if (!isBot && !message.pending) {
      const actions = el('div', 'msg-actions');
      const edit = el('button', '', state.editingIndex === index ? '수정 중' : '수정'); edit.disabled = state.busy;
      edit.addEventListener('click', () => editMessage(index));
      const resend = el('button', '', '다시 보내기'); resend.disabled = state.busy;
      resend.addEventListener('click', () => resendMessage(index));
      actions.append(edit, resend); body.append(actions);
    }
    wrap.append(body);
    return wrap;
  }

  // 답변(추론) 호출에 실은 이미지(Step 8): 모드, 장 수(이름), 상한 때문에 뺀 장 수.
  function describeAnswerImages(mode, info) {
    const dropped = Math.max(0, (info.candidates || 0) - (info.sent || 0));
    let text = `답변 호출 이미지: ${ANSWER_IMAGE_LABELS[mode] || mode}`;
    if (info.sent) text += ` ${info.sent}장` + (info.names?.length && info.names.length <= 4 ? ` (${info.names.join(', ')})` : '');
    if (dropped) text += ` · 상한 때문에 ${dropped}장 제외`;
    return text;
  }

  // 이 답을 어떤 방식으로 만들었는지. 비전 호출도 답변 호출 이미지도 없었던 기본 설정의 답변(일반 대화)에는 표시하지 않는다.
  function describeProcessing(meta) {
    if (!meta || !IMAGE_MODE_LABELS[meta.imageMode]) return '';
    const vision = meta.vision || {};
    const calls = (vision.ocrCalls || 0) + (vision.groundingCalls || 0);
    const answerImages = meta.answerImages || null;
    // 기본이 아닌 모드(끔·전체)를 골랐으면 이미지가 없어도 적는다 — 방식을 바꿔 가며 비교할 때 어느 답이 어느 모드였는지 보이게.
    const showAnswerImages = !!answerImages && (answerImages.sent > 0 || answerImages.candidates > 0 || (!!meta.answerImageMode && meta.answerImageMode !== 'uploads'));
    // 추론 제어(Step 6)의 조치가 있었으면 일반 대화 답변에도 적는다.
    const answerActions = (vision.answerReasoningForced || 0) + (vision.answerReasoningStops || 0);
    if (meta.imageMode !== 'tile' && !calls && !showAnswerImages && !answerActions && !vision.reasoningTokens) return '';
    const parts = [`이미지 처리: ${IMAGE_MODE_LABELS[meta.imageMode]}`];
    if (showAnswerImages) parts.push(describeAnswerImages(meta.answerImageMode, answerImages));
    if (vision.tiles) parts.push(`타일 ${vision.tiles}장${vision.blankTiles ? ` (빈 타일 ${vision.blankTiles}장 제외)` : ''}`);
    // 호출 종류별로 추론을 끄고 보냈는지, 출력 상한에 닿아 끊긴 호출이 있었는지(끊긴 호출은 다시 보내지 않는다),
    // 추론을 끊고 답으로 넘기거나(소프트) 끝내 중단한(하드) 호출이 있었는지(Step 6)
    const detail = (kind, stops, forced, runaways) => {
      const notes = [];
      const off = meta.thinkingDisabled?.[kind];
      if (typeof off === 'boolean') notes.push(off ? '추론 끔' : '추론 끄지 않음');
      if (stops) notes.push(`출력 상한${meta.visionMaxTokens ? ` ${Number(meta.visionMaxTokens).toLocaleString()}토큰` : ''} 도달 ${stops}회`);
      if (forced) notes.push(`추론을 끊고 답으로 넘김 ${forced}회${reasoningWhere(meta.reasoningActions, kind, false)}`);
      if (runaways) notes.push(`추론이 끝나지 않아 중단 ${runaways}회${reasoningWhere(meta.reasoningActions, kind, true)}`);
      return notes.length ? ` (${notes.join(', ')})` : '';
    };
    if (answerActions) parts.push(`답변 호출 ${vision.answerCalls || 0}회${detail('answer', 0, vision.answerReasoningForced, vision.answerReasoningStops)}`);
    if (vision.ocrCalls) parts.push(`전사 호출 ${vision.ocrCalls}회${detail('ocr', vision.ocrLengthStops, vision.ocrReasoningForced, vision.ocrReasoningStops)}`);
    if (vision.groundingCalls) parts.push(`위치 확인 호출 ${vision.groundingCalls}회${detail('grounding', vision.groundingLengthStops, vision.groundingReasoningForced, vision.groundingReasoningStops)}`);
    if (!calls) parts.push('이번 턴에는 전사·위치 확인 호출 없음');
    if (vision.reasoningTokens) parts.push(`추론 ${Number(vision.reasoningTokens).toLocaleString()}토큰`);
    if (typeof meta.elapsedMs === 'number') parts.push(`${(meta.elapsedMs / 1000).toFixed(1)}초`);
    return parts.join(' · ');
  }

  // 추론이 끊긴 호출이 어느 타일(이미지)이었고 왜였는지: " — 반복: r1c1 · 예산: r4c4, r4c5" (Step 6)
  const REASON_LABELS = { repeat: '반복', budget: '예산', empty: '빈 답', rejected: '이어 쓰기 거절' };
  const shortImage = (name) => { const text = String(name || ''); const tile = text.split(' · tile ')[1]; return tile || text.split(' · ').pop() || ''; };
  function reasoningWhere(actions, kind, stopped) {
    const groups = {};
    (actions || []).filter((item) => item.kind === kind && !!item.stopped === stopped).forEach((item) => {
      (groups[item.reason] = groups[item.reason] || []).push(shortImage(item.image));
    });
    const texts = Object.entries(groups).map(([reason, names]) => {
      const named = names.filter(Boolean);
      const shown = named.slice(0, 6).join(', ') + (named.length > 6 ? ` 외 ${named.length - 6}` : '');
      return `${REASON_LABELS[reason] || reason}${shown ? `: ${shown}` : ''}`;
    });
    return texts.length ? ` — ${texts.join(' · ')}` : '';
  }

  // ------------------------------------------------------------------ 턴 과정(트레이스) 창
  const fmtMs = (ms) => ms < 1000 ? `${Math.round(ms)}ms` : ms < 60000 ? `${(ms / 1000).toFixed(1)}s` : `${Math.floor(ms / 60000)}분 ${((ms % 60000) / 1000).toFixed(0)}초`;
  const isPlainObject = (value) => value !== null && typeof value === 'object' && !Array.isArray(value);

  async function openTrace(traceId) {
    state.trace.id = traceId; state.trace.doc = null; state.trace.open = new Set(); state.trace.sections = new Map(); state.trace.collapsed = new Set();
    els.traceDownload.href = `/api/traces/${encodeURIComponent(traceId)}?download=1`;
    els.traceDownload.download = `trace-${traceId}.json`;
    els.traceSummary.replaceChildren(); els.traceBody.replaceChildren(el('div', 'trace-empty', '불러오는 중…'));
    els.traceStatus.textContent = '';
    if (!els.traceDialog.open) els.traceDialog.showModal();
    await loadTrace();
  }

  async function loadTrace() {
    const id = state.trace.id;
    if (!id) return;
    try {
      const doc = await request(`/api/traces/${encodeURIComponent(id)}`);
      if (state.trace.id !== id) return;                       // 그새 다른 트레이스를 열었다
      state.trace.doc = doc;
      renderTrace(doc);
    } catch (error) {
      els.traceBody.replaceChildren(el('div', 'trace-empty', `트레이스를 불러오지 못했습니다: ${error.message}`));
      stopTracePolling();
      return;
    }
    // 진행 중인 턴은 잠시마다 다시 읽어 "진행 중 · 경과 n초"를 갱신한다. 서버가 "살아 있다"고 한 턴만.
    if (state.trace.doc.status === 'running' && state.trace.doc.live !== false && els.traceDialog.open) {
      stopTracePolling();
      state.trace.timer = setTimeout(() => void loadTrace(), 1000);
    } else stopTracePolling();
  }

  function stopTracePolling() { if (state.trace.timer) clearTimeout(state.trace.timer); state.trace.timer = 0; }
  function closeTrace() { stopTracePolling(); state.trace.id = ''; if (els.traceDialog.open) els.traceDialog.close(); }

  function renderTrace(doc) {
    // "진행 중"은 서버가 지금 붙들고 있는 턴(live)일 때만 그렇게 부른다. DB에만 running으로 남은 기록(서버 재시작 등)은
    // 마지막 기록 시점에서 멈춘 것으로 보여 준다 — 시간이 계속 올라가면 살아 있는 것으로 오해한다.
    const stale = doc.status === 'running' && doc.live === false;
    const status = stale ? '기록 중단' : (TRACE_STATUS[doc.status] || doc.status);
    // 살아 있는 턴의 경과는 시작 시각부터 지금까지로 센다. 문서의 elapsedMs는 마지막 저장 시점의 값이라,
    // 새 이벤트가 없으면(모델이 한 호출을 몇 분째 붙들고 있으면) 멈춰 보인다.
    const live = doc.status === 'running' && !stale;
    const now = live ? Math.max(doc.elapsedMs || 0, Date.now() - (doc.createdAt || Date.now())) : (doc.elapsedMs || 0);
    const staleNote = stale ? ` · 마지막 기록 ${fmtMs(Math.max(0, Date.now() - (doc.updatedAt || doc.createdAt || Date.now())))} 전 · 서버에서 끝났는지 알 수 없습니다` : '';
    els.traceStatus.textContent = `${status} · ${fmtMs(now)}` + (doc.reason ? ` · ${doc.reason}` : '') + staleNote;
    els.traceStatus.dataset.status = stale ? 'interrupted' : doc.status;

    // 요약: 호출 수와 걸린 시간을 한 줄로
    const events = doc.events || [];
    const models = events.filter((event) => event.kind === 'model');
    const counts = {};
    models.forEach((event) => { const kind = MODEL_KIND_LABELS[event.data?.kind] || event.data?.kind || '모델'; counts[kind] = (counts[kind] || 0) + 1; });
    const modelTime = models.reduce((sum, event) => sum + (event.elapsedMs || 0), 0);
    const tokens = models.reduce((sum, event) => sum + (event.data?.completionTokens || 0), 0);
    const parts = [
      `모델 호출 ${models.length}회` + (models.length ? ` (${Object.entries(counts).map(([kind, count]) => `${kind} ${count}`).join(', ')})` : ''),
      `도구 실행 ${events.filter((event) => event.kind === 'tool' && event.data?.name && event.data?.arguments).length}회`,
      `모델 대기 합계 ${fmtMs(modelTime)}`,
    ];
    if (tokens) parts.push(`출력 토큰 합계 ${tokens.toLocaleString()}`);
    // 턴 합계(왜 느렸나를 호출 줄을 훑지 않고): 입력·추론 토큰, 추론을 끊은 호출, 가장 오래 걸린 호출
    const promptTokens = models.reduce((sum, event) => sum + (event.data?.promptTokens || 0), 0);
    const reasoningTokens = models.reduce((sum, event) => sum + (event.data?.reasoningTokens || 0), 0);
    if (promptTokens) parts.push(`입력 토큰 합계 ${promptTokens.toLocaleString()}`);
    if (reasoningTokens) parts.push(`추론 토큰 합계 ${reasoningTokens.toLocaleString()}`);
    const cut = models.filter((event) => event.data?.forced || event.data?.runaway);
    if (cut.length) {
      const by = {};
      cut.forEach((event) => { const reason = event.data.forced || String(event.data.runaway).split(':')[0]; by[reason] = (by[reason] || 0) + 1; });
      parts.push(`추론 끊음 ${cut.length}건 (${Object.entries(by).map(([reason, count]) => `${REASON_LABELS[reason] || reason} ${count}`).join(' · ')})`);
    }
    const failed = events.filter((event) => event.status === 'failed' || event.status === 'cancelled').length;
    if (failed) parts.push(`실패·취소 ${failed}건`);
    els.traceSummary.replaceChildren(el('span', '', parts.join(' · ')));
    const slow = models.filter((event) => event.elapsedMs > 0).sort((a, b) => b.elapsedMs - a.elapsedMs).slice(0, 3);
    if (models.length > 3 && slow.length) {
      els.traceSummary.append(el('span', '', '가장 오래 걸린 호출: ' + slow.map((event) =>
        `${shortImage(event.data?.images?.[0]?.name) || event.label} ${fmtMs(event.elapsedMs)}${event.data?.reasoningTokens ? ` (추론 ${Number(event.data.reasoningTokens).toLocaleString()})` : ''}`).join(' · ')));
    }
    els.traceSummary.append(el('span', 'trace-hint', '행을 누르면 세부가 열리고, 도구·전사 묶음은 접힙니다(세부는 "세부" 버튼).'));

    // 타임라인: 트리로 그린다 — 자식(모델 호출)은 시작 시각과 무관하게 부모(도구·전사) 바로 아래에 들여 쓴다.
    // (타일 20장은 한꺼번에 시작해 순서대로 호출되므로, 시작 시각순으로 펼치면 자식이 엉뚱한 부모 밑에 보인다.)
    const known = new Set(events.map((event) => event.id));
    const children = new Map();
    for (const event of events) {
      const key = event.parent && known.has(event.parent) ? event.parent : 0;
      if (!children.has(key)) children.set(key, []);
      children.get(key).push(event);
    }
    const body = el('div', 'trace-list');
    const spanKinds = new Set(['model', 'tool', 'ocr', 'preprocess']);
    // 도구 실행은 한 턴에 여러 번이고 제목이 같기 쉽다(같은 이미지를 문장만 바꿔 다시 부름) → 번호와 작업 문장으로 구분한다.
    const executions = events.filter((event) => event.kind === 'tool' && event.data?.name && event.data?.arguments);
    const titleOf = (event) => {
      const index = executions.indexOf(event);
      if (index < 0) return event.label;
      const args = event.data.arguments || {};
      const what = args.task || args.query || args.name || '';
      return `도구 실행 #${index + 1} · ${event.data.name}${what ? ` — ${String(what).slice(0, 60)}${String(what).length > 60 ? '…' : ''}` : ''}`;
    };
    const emit = (event, depth) => {
      const cutShort = event.kind === 'model' && (event.data?.forced || event.data?.runaway);      // 추론을 끊은 호출(Step 6)
      const row = el('div', `trace-row-item k-${event.kind} s-${event.status}${state.trace.open.has(event.id) ? ' open' : ''}${cutShort ? ' cut' : ''}`);
      row.style.setProperty('--depth', depth);
      const head = el('button', 'trace-head'); head.type = 'button';
      head.append(el('span', 'trace-time', `+${fmtMs(event.startedMs)}`));
      const running = event.status === 'running';
      const own = children.get(event.id) || [];
      // 자식이 있는 행(도구·전사)은 행을 누르면 가지를 접고 펼친다(타일 20장이면 한 도구 아래 60줄이 넘는다).
      // 세부(인자·결과)는 오른쪽 "세부" 버튼으로 연다. 자식이 없는 행은 행을 누르면 세부가 열린다.
      const folded = state.trace.collapsed.has(event.id);
      const fold = el('span', `trace-fold${own.length ? '' : ' none'}`, own.length ? (folded ? '▸' : '▾') : '');
      head.append(fold);
      const toggleDetail = () => { state.trace.open.has(event.id) ? state.trace.open.delete(event.id) : state.trace.open.add(event.id); renderTrace(state.trace.doc); };
      const toggleFold = () => { folded ? state.trace.collapsed.delete(event.id) : state.trace.collapsed.add(event.id); renderTrace(state.trace.doc); };
      head.title = own.length ? (folded ? `펼치기 (${own.length}개)` : '접기') : '세부 보기';
      const firstCall = own.find((child) => child.kind === 'model');
      // 전사 타일은 한꺼번에 시작해 차례를 기다린다 → 첫 모델 호출 전까지는 "대기", 끝난 뒤에는 대기 시간을 따로 보인다.
      const wait = firstCall ? firstCall.startedMs - event.startedMs : (running ? now - event.startedMs : 0);
      let duration = '';
      if (running) duration = (stale ? '기록 중단 · ' : event.kind === 'ocr' && !firstCall ? '차례 대기 중 · ' : '진행 중 · ') + fmtMs(Math.max(0, now - event.startedMs))
        + (event.data?.reasoningStage === 'reasoning' && event.data.reasoningTokens ? ` · 추론 ${Number(event.data.reasoningTokens).toLocaleString()}토큰` : '');
      else if (cutShort) {
        const reason = event.data.forced || String(event.data.runaway).split(':')[0];
        duration = `${fmtMs(event.elapsedMs || 0)} · 추론 끊음(${REASON_LABELS[reason] || reason}${event.data.runaway ? ' → 중단' : ''})`
          + (event.data.forcedCycle ? ` · “${String(event.data.forcedCycle).slice(0, 48)}”` : '');
      }
      else if (spanKinds.has(event.kind) || event.elapsedMs > 0) duration = fmtMs(event.elapsedMs || 0) + (event.kind === 'ocr' && wait > 100 ? ` (대기 ${fmtMs(wait)})` : '');
      head.append(el('span', 'trace-kind', TRACE_KINDS[event.kind] || event.kind));
      head.append(el('span', 'trace-label', titleOf(event) + (folded ? ` (+${own.length})` : '')));
      head.append(el('span', `trace-dur${running ? ' live' : ''}`, duration));
      head.append(el('span', 'trace-state', event.status === 'done' ? '' : (TRACE_STATUS[event.status] || event.status)));
      if (own.length) {
        const detail = el('span', `trace-detail-btn${state.trace.open.has(event.id) ? ' on' : ''}`, '세부');
        detail.title = '이 실행의 인자·결과';
        detail.addEventListener('click', (click) => { click.stopPropagation(); toggleDetail(); });
        head.append(detail);
        head.addEventListener('click', toggleFold);
      } else {
        head.append(el('span', 'trace-detail-btn none', ''));
        head.addEventListener('click', toggleDetail);
      }
      row.append(head);
      if (state.trace.open.has(event.id)) row.append(traceDetail(event));
      body.append(row);
      if (!folded) own.forEach((child) => emit(child, depth + 1));
    };
    (children.get(0) || []).forEach((event) => emit(event, 0));
    if (!events.length) body.append(el('div', 'trace-empty', '아직 기록된 이벤트가 없습니다.'));
    const scrolled = els.traceBody.scrollTop;
    els.traceBody.replaceChildren(body);
    els.traceBody.scrollTop = scrolled;
  }

  // 이벤트 하나의 세부 내용. 모델 출력은 전부 textContent로만 넣는다.
  let detailEvent = 0;      // 지금 세부를 그리는 이벤트 id — 접이식의 열림 상태를 기억하는 키에 쓴다
  function traceDetail(event) {
    detailEvent = event.id;
    const box = el('div', 'trace-detail');
    const data = event.data || {};
    if (event.kind === 'model') {
      const chips = [
        ['종류', MODEL_KIND_LABELS[data.kind] || data.kind], ['모델', data.model], ['temperature', data.temperature],
        ['추론 끄기', data.thinkingControl === false ? '서버가 지원하지 않음' : (data.disableThinking ? '예' : '아니오')],
        ['출력 상한', data.maxTokens], ['종료 사유', data.finishReason], ['입력 토큰', data.promptTokens], ['출력 토큰', data.completionTokens],
        // 추론 제어(Step 6): 예산, 센 추론 토큰·시간, 추론을 끊고 답으로 넘겼는지(소프트), 끝내 중단했는지(하드)
        ['추론 예산', data.reasoningBudget], ['추론 토큰', data.reasoningTokens],
        ['추론 시간', typeof data.reasoningSeconds === 'number' ? `${data.reasoningSeconds}s` : undefined],
        ['추론 조치', data.forced ? `${data.forced === 'repeat' ? '반복' : '예산 초과'} → 추론을 끊고 답으로` : undefined],
        ['되풀이된 줄', data.forcedCycle],
        ['중단', data.runaway],
        ['도구 제공', (data.tools || []).join(', ') || '없음'],
      ];
      box.append(chipRow(chips));
      if (data.error) box.append(el('div', 'trace-error', data.error));
      if (data.reason) box.append(el('div', 'trace-error', data.reason));
      if (data.images?.length) box.append(section('보낸 이미지', imageTable(data.images), true));
      box.append(section(`보낸 메시지 ${data.messages?.length || 0}개`, messageList(data.messages || [], (data.images || []).length), false));
      if (data.text !== undefined) box.append(section(`응답 본문 (${(data.textChars || 0).toLocaleString()}자)`, pre(data.text || '(비어 있음)'), true));
      if (data.reasoning) box.append(section(`추론 (${(data.reasoningChars || 0).toLocaleString()}자)`, pre(data.reasoning), false));
      if (data.toolCalls?.length) box.append(section('도구 호출', pre(JSON.stringify(data.toolCalls, null, 2)), true));
      return box;
    }
    if (event.kind === 'tool' && data.arguments) {
      box.append(chipRow([['도구', data.name], ['순서', data.step]]));
      box.append(section('인자', pre(JSON.stringify(data.arguments, null, 2)), true));
      if (data.result !== undefined) box.append(section(`결과 (${(data.resultChars || 0).toLocaleString()}자)`, pre(data.result), false));
      if (data.error) box.append(el('div', 'trace-error', data.error));
      if (data.reason) box.append(el('div', 'trace-error', data.reason));
      return box;
    }
    for (const [key, value] of Object.entries(data)) box.append(traceValue(key, value));
    return box;
  }

  function traceValue(key, value) {
    if (Array.isArray(value)) {
      if (value.length && value.every(isPlainObject)) {
        if (key === 'messages') return section(`${key} (${value.length})`, messageList(value), false);
        if (key === 'images') return section(`${key} (${value.length})`, imageTable(value), true);
        return section(`${key} (${value.length})`, objectTable(value), true);
      }
      return section(key, pre(JSON.stringify(value, null, 2)), value.length <= 12);
    }
    if (isPlainObject(value)) return section(key, chipRow(Object.entries(value)), true);
    if (typeof value === 'string' && (value.length > 160 || value.includes('\n'))) return section(`${key} (${value.length.toLocaleString()}자)`, pre(value), key === 'text' || key === 'question');
    return chipRow([[key, value]]);
  }

  function chipRow(pairs) {
    const row = el('div', 'trace-chips');
    for (const [key, value] of pairs) {
      if (value === undefined || value === null || value === '') continue;
      const chip = el('span', 'trace-chip');
      chip.append(el('b', '', `${key} `), el('span', '', typeof value === 'object' ? JSON.stringify(value) : String(value)));
      row.append(chip);
    }
    return row;
  }

  // 접이식. 진행 중인 턴은 1초마다 다시 그리므로, 사용자가 펼치거나 접은 상태를 (이벤트, 제목) 키로 기억해 되살린다.
  function section(title, content, open) {
    const key = `${detailEvent}:${title}`;
    const details = el('details', 'trace-section');
    const remembered = state.trace.sections.get(key);
    details.open = remembered === undefined ? !!open : remembered;
    details.addEventListener('toggle', () => state.trace.sections.set(key, details.open));
    details.append(el('summary', '', title), content);
    return details;
  }

  function pre(text) { const node = el('pre', 'trace-pre'); node.textContent = String(text ?? ''); return node; }

  // imageCount: 이 호출에 실린 이미지 수. 닻 표시(imagesAnchor)는 이미지가 없어도 붙으므로 실제 이미지가 있을 때만 적는다.
  function messageList(messages, imageCount = 0) {
    const list = el('div', 'trace-messages');
    messages.forEach((message, index) => {
      const title = `${index + 1}. ${message.role}${message.name ? ` (${message.name})` : ''} · ${(message.chars || 0).toLocaleString()}자`
        + (message.imagesAnchor && imageCount ? ` · 이미지 ${imageCount}장 첨부` : '') + (message.toolCalls ? ` · 도구 호출 ${message.toolCalls.length}건` : '');
      const content = el('div');
      if (message.content) content.append(pre(message.content));
      if (message.toolCalls) content.append(pre(JSON.stringify(message.toolCalls, null, 2)));
      list.append(section(title, content, index === messages.length - 1 && message.role !== 'system'));
    });
    return list;
  }

  function imageTable(images) {
    const rows = images.map((image) => ({
      이름: image.name, 크기: image.width && image.height ? `${image.width}×${image.height}` : '', 바이트: image.bytes,
      타일: image.tile ? `r${image.tile[0]}c${image.tile[1]}` : '', 영역: image.sourceBox ? image.sourceBox.join(', ') : '',
      // 뷰어는 모달 뒤에 있으므로 창을 닫고 연다(타일은 저장된 첨부가 아니라 쪽·이미지 전체가 열린다).
      _open: Number.isInteger(image.attachmentId) ? () => { closeTrace(); openArtifact({ name: image.name, title: image.name, mime: image.mime || 'image/png', view: 'image', attachmentId: image.attachmentId }); } : null,
    }));
    return objectTable(rows);
  }

  function objectTable(rows) {
    const columns = [...new Set(rows.flatMap((row) => Object.keys(row).filter((key) => !key.startsWith('_'))))];
    const wrap = el('div', 'md-table-wrap'), node = el('table', 'md-table trace-table'), head = el('tr'), body = el('tbody');
    columns.forEach((column) => head.append(el('th', '', column)));
    if (rows.some((row) => row._open)) head.append(el('th', '', ''));
    const thead = el('thead'); thead.append(head); node.append(thead);
    for (const row of rows) {
      const tr = el('tr');
      for (const column of columns) {
        const value = row[column];
        const text = value === undefined || value === null ? '' : typeof value === 'object' ? JSON.stringify(value) : String(value);
        const cell = el('td', '', text.length > 200 ? `${text.slice(0, 200)}…` : text);
        if (text.length > 200) cell.title = text;
        tr.append(cell);
      }
      if (row._open) { const open = el('button', 'link-btn', '보기'); open.type = 'button'; open.addEventListener('click', row._open); const cell = el('td'); cell.append(open); tr.append(cell); }
      else if (rows.some((item) => item._open)) tr.append(el('td'));
      body.append(tr);
    }
    node.append(body); wrap.append(node);
    return wrap;
  }

  // ------------------------------------------------------------------ 결과 뷰어 (이미지 + bbox 오버레이)
  const view = { frame: null, width: 0, zoom: 1, fit: 1, pan: null, resizing: false };

  function openArtifact(artifact) {
    els.viewerTitle.textContent = artifact.title || artifact.name;
    const boxes = artifact.boxes || [];
    els.viewerMeta.textContent = [artifact.task ? `요청: ${artifact.task}` : '', boxes.length ? `영역 ${boxes.length}개` : ''].filter(Boolean).join(' · ');
    els.viewerDownload.href = attachmentUrl(artifact.attachmentId, true);
    els.viewerDownload.download = artifact.name;
    els.viewerStage.replaceChildren(); els.viewerRegions.replaceChildren();
    view.frame = null;

    const frame = el('div', 'viewer-frame');
    const image = el('img'); image.alt = artifact.title || artifact.name; image.draggable = false;
    image.addEventListener('load', () => {
      view.frame = frame; view.width = image.naturalWidth;
      view.fit = Math.min(1, (els.viewerStage.clientWidth - 24) / Math.max(1, image.naturalWidth));
      setZoom(view.fit);
    });
    image.addEventListener('error', () => { els.viewerStage.replaceChildren(el('div', 'viewer-message', '이미지를 불러오지 못했습니다. 세션이 삭제됐을 수 있습니다.')); });
    image.src = attachmentUrl(artifact.attachmentId);
    frame.append(image);

    const activate = (index) => {
      frame.querySelectorAll('.box').forEach((node, i) => node.classList.toggle('active', i === index));
      els.viewerRegions.querySelectorAll('.region').forEach((node, i) => node.classList.toggle('active', i === index));
    };
    if (artifact.text) els.viewerRegions.append(el('div', 'regions-note', artifact.text));
    boxes.forEach((box, index) => {
      const type = TYPE_LABELS[box.type] ? box.type : 'other';
      // 좌표는 이미지 대비 분수(0~1) → %로 두면 확대/축소와 무관하게 정확히 따라붙는다.
      const overlay = el('div', `box t-${type}`);
      Object.assign(overlay.style, { left: `${box.x * 100}%`, top: `${box.y * 100}%`, width: `${box.w * 100}%`, height: `${box.h * 100}%` });
      overlay.title = box.label || TYPE_LABELS[type];
      if (box.label) overlay.append(el('span', 'box-label', box.label));
      overlay.addEventListener('click', (event) => { event.stopPropagation(); activate(index); els.viewerRegions.querySelectorAll('.region')[index]?.scrollIntoView({ block: 'nearest' }); });
      frame.append(overlay);

      const row = el('div', `region t-${type}`);
      row.append(el('span', 'region-type', TYPE_LABELS[type]), el('span', 'region-label', box.label || '(라벨 없음)'),
        el('span', 'region-conf', typeof box.confidence === 'number' ? `${Math.round(box.confidence * 100)}%` : ''));
      row.addEventListener('click', () => { activate(index); overlay.scrollIntoView({ block: 'center', inline: 'center', behavior: 'smooth' }); });
      els.viewerRegions.append(row);
    });
    els.viewerStage.append(frame);
    els.viewer.hidden = false; els.shell.classList.add('viewer-open');
  }

  function closeViewer() { view.frame = null; els.viewer.hidden = true; els.shell.classList.remove('viewer-open'); }

  function setZoom(value, clientX, clientY) {
    if (!view.frame || !view.width) return;
    const stage = els.viewerStage, previous = view.zoom, next = Math.max(0.1, Math.min(8, value));
    const rect = stage.getBoundingClientRect();
    const px = (clientX ?? rect.left + rect.width / 2) - rect.left, py = (clientY ?? rect.top + rect.height / 2) - rect.top;
    const contentX = stage.scrollLeft + px, contentY = stage.scrollTop + py;
    view.zoom = next;
    view.frame.style.width = `${view.width * next}px`;
    els.zoomReset.textContent = `${Math.round(next * 100)}%`;
    stage.scrollLeft = contentX * (next / previous) - px;      // 커서 아래 지점이 그대로 머물도록
    stage.scrollTop = contentY * (next / previous) - py;
  }

  // ------------------------------------------------------------------ 마크다운(경량): 표·목록·제목·코드·굵게/기울임/링크
  function renderMarkdown(container, source) {
    const lines = String(source || '').replace(/\r\n/g, '\n').split('\n');
    const isTableRow = (line) => /^\s*\|.*\|\s*$/.test(line || '');
    const cells = (line) => line.trim().replace(/^\||\|$/g, '').split(/(?<!\\)\|/).map((cell) => cell.trim().replace(/\\\|/g, '|'));
    const isDivider = (line) => isTableRow(line) && cells(line).every((cell) => /^:?-{3,}:?$/.test(cell));
    let index = 0, list = null;
    while (index < lines.length) {
      const line = lines[index];
      const fence = line.match(/^\s*```\s*([\w+-]*)\s*$/);
      if (fence) {
        list = null;
        const code = [];
        for (index++; index < lines.length && !/^\s*```\s*$/.test(lines[index]); index++) code.push(lines[index]);
        index++;
        container.append(codeBlock(code.join('\n'), fence[1]));
        continue;
      }
      if (!line.trim()) { list = null; index++; continue; }
      if (isTableRow(line) && isDivider(lines[index + 1])) {
        list = null;
        const rows = [line];
        for (index += 2; index < lines.length && isTableRow(lines[index]); index++) rows.push(lines[index]);
        container.append(table(rows.map(cells)));
        continue;
      }
      const heading = line.match(/^(#{1,6})\s+(.*)$/);
      if (heading) { list = null; const node = el('div', `md-h l${heading[1].length}`); inline(node, heading[2]); container.append(node); index++; continue; }
      const item = line.match(/^\s*(?:(\d+)[.)]|[-*•])\s+(.*)$/);
      if (item) {
        const tag = item[1] ? 'ol' : 'ul';
        if (!list || list.tagName.toLowerCase() !== tag) { list = el(tag, 'md-list'); container.append(list); }
        const li = el('li'); inline(li, item[2]); list.append(li); index++; continue;
      }
      list = null;
      const paragraph = [line];
      for (index++; index < lines.length && lines[index].trim() && !/^\s*```/.test(lines[index]) && !/^#{1,6}\s/.test(lines[index]) && !/^\s*(?:\d+[.)]|[-*•])\s+/.test(lines[index]) && !(isTableRow(lines[index]) && isDivider(lines[index + 1])); index++) paragraph.push(lines[index]);
      const node = el('div', 'md-p'); inline(node, paragraph.join('\n')); container.append(node);
    }
  }

  function inline(node, text) {
    const pattern = /(`[^`\n]+`)|(\[[^\]\n]+\]\(https?:\/\/[^\s)]+\))|(\*\*[^*\n]+\*\*)|(\*[^*\n]+\*)/g;
    let last = 0, match;
    while ((match = pattern.exec(text))) {
      if (match.index > last) node.append(text.slice(last, match.index));
      const token = match[0];
      if (match[1]) node.append(el('code', 'md-code-inline', token.slice(1, -1)));
      else if (match[2]) {
        const [, label, url] = token.match(/^\[([^\]]+)\]\((.+)\)$/);
        const link = el('span', 'md-link', label); link.title = url;
        link.addEventListener('click', () => window.open(url, '_blank', 'noopener,noreferrer'));
        node.append(link);
      } else if (match[3]) node.append(el('strong', '', token.slice(2, -2)));
      else node.append(el('em', '', token.slice(1, -1)));
      last = pattern.lastIndex;
    }
    if (last < text.length) node.append(text.slice(last));
  }

  function table(rows) {
    const wrap = el('div', 'md-table-wrap'), node = el('table', 'md-table'), head = el('thead'), body = el('tbody');
    rows.forEach((row, rowIndex) => {
      const tr = el('tr');
      row.forEach((value) => { const cell = el(rowIndex ? 'td' : 'th'); inline(cell, value); tr.append(cell); });
      (rowIndex ? body : head).append(tr);
    });
    node.append(head, body); wrap.append(node);
    return wrap;
  }

  function codeBlock(code, language) {
    const wrap = el('div', 'md-pre'), bar = el('div', 'md-pre-bar'), copy = el('button', '', '복사');
    copy.type = 'button';
    copy.addEventListener('click', () => { navigator.clipboard?.writeText(code).then(() => { copy.textContent = '복사됨'; setTimeout(() => { copy.textContent = '복사'; }, 1400); }).catch(() => {}); });
    bar.append(el('span', '', language || 'text'), copy);
    const pre = el('pre'); pre.append(el('code', '', code));
    wrap.append(bar, pre);
    return wrap;
  }

  // ------------------------------------------------------------------ 기타 UI
  function autoGrow() { els.prompt.style.height = 'auto'; els.prompt.style.height = `${Math.min(180, Math.max(28, els.prompt.scrollHeight))}px`; }

  function applyTheme(theme) {
    if (theme === 'light') document.documentElement.dataset.theme = 'light'; else delete document.documentElement.dataset.theme;
    els.themeToggle.textContent = theme === 'light' ? '☾ 어둡게' : '☀ 밝게';
    store.set('theme', theme);
  }

  // ------------------------------------------------------------------ 이벤트 연결
  els.newChat.addEventListener('click', newChat);
  els.selectSessions.addEventListener('click', toggleSelecting);
  els.refreshSessions.addEventListener('click', () => loadSessions().catch((error) => alert(error.message)));
  els.deleteSelected.addEventListener('click', () => deleteSelected().catch((error) => alert(error.message)));
  els.deleteAll.addEventListener('click', () => deleteAll().catch((error) => alert(error.message)));
  els.openSettings.addEventListener('click', openSettings);
  els.providerSelect.addEventListener('change', syncSettingsForm);
  els.disableThinking.addEventListener('change', syncCallThinking);
  els.settingsTest.addEventListener('click', () => void testFromSettings());
  els.saveSettings.addEventListener('click', saveSettings);
  els.testConnection.addEventListener('click', () => void testConnection());
  els.modelSelect.addEventListener('change', () => { state.model = els.modelSelect.value; persistSettings(); });
  els.themeToggle.addEventListener('click', () => applyTheme(document.documentElement.dataset.theme === 'light' ? 'dark' : 'light'));
  els.attach.addEventListener('click', () => els.fileInput.click());
  els.fileInput.addEventListener('change', async () => { await addFiles([...els.fileInput.files]); els.fileInput.value = ''; });
  els.send.addEventListener('click', () => { if (state.busy) state.abort?.abort(); else void send(); });
  els.prompt.addEventListener('input', autoGrow);
  els.prompt.addEventListener('keydown', (event) => { if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) { event.preventDefault(); void send(); } });
  ['dragenter', 'dragover'].forEach((type) => els.dropZone.addEventListener(type, (event) => { event.preventDefault(); els.dropZone.classList.add('dragging'); }));
  ['dragleave', 'drop'].forEach((type) => els.dropZone.addEventListener(type, (event) => { event.preventDefault(); els.dropZone.classList.remove('dragging'); }));
  els.dropZone.addEventListener('drop', (event) => void addFiles([...event.dataTransfer.files]));
  // 입력창 밖에 떨어뜨려도 브라우저가 파일을 열어 버리지 않게 한다.
  ['dragover', 'drop'].forEach((type) => window.addEventListener(type, (event) => event.preventDefault()));

  els.traceClose.addEventListener('click', closeTrace);
  els.traceRefresh.addEventListener('click', () => void loadTrace());
  // 자식이 있는 이벤트(도구·전사 묶음)를 한꺼번에 접거나 펼친다.
  els.traceFoldAll.addEventListener('click', () => {
    const events = state.trace.doc?.events || [];
    const parents = new Set(events.map((event) => event.parent).filter(Boolean));
    state.trace.collapsed = new Set(events.filter((event) => parents.has(event.id)).map((event) => event.id));
    if (state.trace.doc) renderTrace(state.trace.doc);
  });
  els.traceUnfoldAll.addEventListener('click', () => { state.trace.collapsed = new Set(); if (state.trace.doc) renderTrace(state.trace.doc); });
  els.traceDialog.addEventListener('close', stopTracePolling);
  els.traceDialog.addEventListener('cancel', stopTracePolling);

  els.viewerClose.addEventListener('click', closeViewer);
  els.zoomIn.addEventListener('click', () => setZoom(view.zoom * 1.25));
  els.zoomOut.addEventListener('click', () => setZoom(view.zoom / 1.25));
  els.zoomReset.addEventListener('click', () => setZoom(view.fit));
  els.toggleLabels.addEventListener('click', () => els.viewerStage.classList.toggle('labels-off'));
  els.viewerStage.addEventListener('wheel', (event) => { if (!view.frame) return; event.preventDefault(); setZoom(view.zoom * (event.deltaY < 0 ? 1.12 : 1 / 1.12), event.clientX, event.clientY); }, { passive: false });
  els.viewerStage.addEventListener('pointerdown', (event) => {
    if (!view.frame || event.button !== 0 || event.target.closest('.box')) return;
    view.pan = { x: event.clientX, y: event.clientY, left: els.viewerStage.scrollLeft, top: els.viewerStage.scrollTop };
    els.viewerStage.setPointerCapture(event.pointerId); els.viewerStage.classList.add('panning');
  });
  els.viewerStage.addEventListener('pointermove', (event) => {
    if (!view.pan) return;
    els.viewerStage.scrollLeft = view.pan.left - (event.clientX - view.pan.x);
    els.viewerStage.scrollTop = view.pan.top - (event.clientY - view.pan.y);
  });
  ['pointerup', 'pointercancel'].forEach((type) => els.viewerStage.addEventListener(type, () => { view.pan = null; els.viewerStage.classList.remove('panning'); }));

  const savedWidth = Number(store.get('viewerWidth', '0'));
  if (savedWidth > 0) els.shell.style.setProperty('--viewer-width', `${savedWidth}px`);
  els.viewerResize.addEventListener('pointerdown', (event) => { if (event.button !== 0) return; view.resizing = true; els.viewerResize.setPointerCapture(event.pointerId); els.viewerResize.classList.add('dragging'); document.body.classList.add('resizing'); event.preventDefault(); });
  els.viewerResize.addEventListener('pointermove', (event) => { if (view.resizing) els.shell.style.setProperty('--viewer-width', `${Math.round(els.shell.getBoundingClientRect().right - event.clientX)}px`); });
  ['pointerup', 'pointercancel'].forEach((type) => els.viewerResize.addEventListener(type, () => {
    if (!view.resizing) return;
    view.resizing = false; els.viewerResize.classList.remove('dragging'); document.body.classList.remove('resizing');
    store.set('viewerWidth', parseInt(els.shell.style.getPropertyValue('--viewer-width'), 10) || '');
  }));

  // ------------------------------------------------------------------ 시작
  applyTheme(document.documentElement.dataset.theme === 'light' ? 'light' : 'dark');
  renderModelBar(); render();
  loadSessions().catch(() => {});
  void loadModels();
  void loadServerDefaults();
})();
