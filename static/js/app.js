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
    // 호출 종류별 추론 끄기. 'true' | 'false' | ''(고른 적 없음 → 서버 기본값)
    disableThinkingGrounding: store.get('disableThinkingGrounding', ''),
    disableThinkingOcr: store.get('disableThinkingOcr', ''),
    serverVision: { disableThinkingGrounding: true, disableThinkingOcr: true, maxTokens: 4096 },
    models: []
  };
  if (!PROVIDER_LABELS[state.provider]) state.provider = LOCAL_PROVIDER;
  if (!IMAGE_MODE_LABELS[state.imageMode]) state.imageMode = '';

  const $ = (id) => document.getElementById(id);
  const els = Object.fromEntries([
    'shell', 'newChat', 'selectSessions', 'refreshSessions', 'sessionList', 'deleteSelected', 'deleteAll',
    'modelSelect', 'testConnection', 'providerBadge', 'themeToggle', 'openSettings', 'messages', 'pendingFiles',
    'dropZone', 'attach', 'prompt', 'send', 'fileInput', 'viewer', 'viewerResize', 'viewerTitle', 'viewerMeta',
    'toggleLabels', 'zoomOut', 'zoomReset', 'zoomIn', 'viewerDownload', 'viewerClose', 'viewerStage', 'viewerRegions',
    'settingsDialog', 'providerSelect', 'baseUrlField', 'baseUrl', 'apiKey', 'apiKeyLabel', 'modelName', 'modelOptions',
    'contextField', 'contextSize', 'disableThinking', 'callThinking', 'disableThinkingGrounding', 'disableThinkingOcr', 'callThinkingHelp',
    'imageMode', 'imageModeHelp', 'settingsTest', 'testDot', 'testText', 'saveSettings'
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
    render(); renderSessions();
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
    store.set('imageMode', state.imageMode);
    store.set('disableThinkingGrounding', state.disableThinkingGrounding); store.set('disableThinkingOcr', state.disableThinkingOcr);
  }

  // 요청에 실을 이미지 처리 방식. 사용자가 고른 적이 없으면 서버 기본값.
  const imageMode = () => state.imageMode || state.serverImageMode;
  // 요청에 실을 호출별 추론 끄기(kind: 'disableThinkingGrounding' | 'disableThinkingOcr'). 고른 적이 없으면 서버 기본값.
  const callThinkingOff = (kind) => state[kind] === '' ? state.serverVision[kind] !== false : state[kind] === 'true';

  async function loadServerDefaults() {
    try {
      const health = await request('/api/health');
      if (IMAGE_MODE_LABELS[health.imageMode]) state.serverImageMode = health.imageMode;
      state.tiling = health.tiling || null;
      if (health.vision) state.serverVision = { ...state.serverVision, ...health.vision };
    } catch { /* 기본값(전체)으로 둔다 */ }
  }

  // "추론 끄기"(모든 호출)가 켜져 있으면 호출별 선택은 적용되지 않는다 → 고른 값은 그대로 두고 잠가서 보여 준다.
  function syncCallThinking() {
    const everything = els.disableThinking.checked;
    els.disableThinkingGrounding.disabled = everything; els.disableThinkingOcr.disabled = everything;
    els.callThinking.classList.toggle('inactive', everything);
    const limit = state.serverVision.maxTokens;
    els.callThinkingHelp.textContent = (everything
      ? '위 항목이 켜져 있어 모든 호출의 추론이 꺼집니다. 위 항목을 끄면 아래 두 선택이 적용됩니다.'
      : '답변 호출만 추론을 쓰고, 위치 확인과 전사는 끌 수 있습니다. 이 둘은 보이는 것을 옮겨 적는 호출이라, 추론을 켜면 같은 생각을 맴돌다 끝나지 않는 일이 있습니다.')
      + (limit > 0
        ? ` 위치 확인·전사 호출은 출력 ${Number(limit).toLocaleString()}토큰에서 끊고 다시 보내지 않습니다(.env의 DOCCHAT_VISION_MAX_TOKENS). 이 둘에 추론을 켜려면 8,000 이상을 권합니다.`
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
        } else {
          item.status = report.resized ? `${report.sourceWidth}×${report.sourceHeight} → ${report.width}×${report.height}로 축소해 전달` : `${report.width}×${report.height} 원본 그대로 전달`;
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
        imageMode: imageMode(), conversationId: state.currentId, stream: true,
        // meta(답변을 어떤 방식으로 처리했는지)도 되돌려 보내야 서버가 대화를 다시 저장할 때 지워지지 않는다.
        messages: history.map(({ role, content, files: sent, artifacts, meta, createdAt }) => ({ role, content, files: sent, artifacts, meta, createdAt })),
        attachments: files.map(({ name, mime, size, base64 }) => ({ name, mime, size, base64 }))
      }, state.abort.signal, (event) => {
        if (event.type === 'conversation') state.currentId = event.conversationId;
        if (event.type === 'progress' && placeholder.activity.at(-1) !== event.message) { placeholder.activity.push(event.message); render(); }
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

  // 이 답을 어떤 이미지 처리 방식으로 만들었는지. 비전 호출이 없었던 전체 모드 답변(일반 대화)에는 표시하지 않는다.
  function describeProcessing(meta) {
    if (!meta || !IMAGE_MODE_LABELS[meta.imageMode]) return '';
    const vision = meta.vision || {};
    const calls = (vision.ocrCalls || 0) + (vision.groundingCalls || 0);
    if (meta.imageMode !== 'tile' && !calls) return '';
    const parts = [`이미지 처리: ${IMAGE_MODE_LABELS[meta.imageMode]}`];
    if (vision.tiles) parts.push(`타일 ${vision.tiles}장${vision.blankTiles ? ` (빈 타일 ${vision.blankTiles}장 제외)` : ''}`);
    // 호출 종류별로 추론을 끄고 보냈는지, 출력 상한에 닿아 끊긴 호출이 있었는지(끊긴 호출은 다시 보내지 않는다)
    const detail = (kind, stops) => {
      const notes = [];
      const off = meta.thinkingDisabled?.[kind];
      if (typeof off === 'boolean') notes.push(off ? '추론 끔' : '추론 끄지 않음');
      if (stops) notes.push(`출력 상한${meta.visionMaxTokens ? ` ${Number(meta.visionMaxTokens).toLocaleString()}토큰` : ''} 도달 ${stops}회`);
      return notes.length ? ` (${notes.join(', ')})` : '';
    };
    if (vision.ocrCalls) parts.push(`전사 호출 ${vision.ocrCalls}회${detail('ocr', vision.ocrLengthStops)}`);
    if (vision.groundingCalls) parts.push(`위치 확인 호출 ${vision.groundingCalls}회${detail('grounding', vision.groundingLengthStops)}`);
    if (!calls) parts.push('이번 턴에는 전사·위치 확인 호출 없음');
    if (typeof meta.elapsedMs === 'number') parts.push(`${(meta.elapsedMs / 1000).toFixed(1)}초`);
    return parts.join(' · ');
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
