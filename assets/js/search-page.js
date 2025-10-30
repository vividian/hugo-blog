(() => {
  const ROOT_SELECTOR = "[data-search-page]";
  const RESULTS_SELECTOR = "#search-results";
  const MESSAGE_SELECTOR = "#search-message";
  const INDEX_URL = "/index.json";
  const MIN_QUERY_LENGTH = 2;
  const FUSE_CDN = "https://cdn.jsdelivr.net/npm/fuse.js@6.6.2/dist/fuse.min.js";

  const root = document.querySelector(ROOT_SELECTOR);
  if (!root) {
    return;
  }

  const resultsContainer = root.querySelector(RESULTS_SELECTOR);
  const messageContainer = root.querySelector(MESSAGE_SELECTOR);

  if (!resultsContainer || !messageContainer) {
    return;
  }

  const state = {
    fusePromise: null,
    fuse: null,
    indexPromise: null,
    index: [],
  };

  const loadFuse = () => {
    if (window.Fuse) {
      return Promise.resolve(window.Fuse);
    }

    if (state.fusePromise) {
      return state.fusePromise;
    }

    state.fusePromise = new Promise((resolve, reject) => {
      const script = document.createElement("script");
      script.src = FUSE_CDN;
      script.async = true;
      script.onload = () => (window.Fuse ? resolve(window.Fuse) : reject(new Error("Fuse.js failed to load.")));
      script.onerror = () => reject(new Error("Fuse.js failed to load."));
      document.head.appendChild(script);
    });

    return state.fusePromise;
  };

  const loadIndex = () => {
    if (state.indexPromise) {
      return state.indexPromise;
    }

    state.indexPromise = fetch(INDEX_URL, { credentials: "same-origin" })
      .then((response) => {
        if (!response.ok) {
          throw new Error(`Failed to load search index. Status: ${response.status}`);
        }
        return response.json();
      })
      .then((payload) => {
        if (!Array.isArray(payload)) {
          throw new Error("Unexpected search index format.");
        }
        state.index = payload;
        return payload;
      })
      .catch((error) => {
        console.error(error);
        state.index = [];
        return [];
      });

    return state.indexPromise;
  };

  const ensureFuse = () => {
    if (state.fuse) {
      return Promise.resolve(state.fuse);
    }

    return Promise.all([loadFuse(), loadIndex()]).then(([Fuse]) => {
      state.fuse = new Fuse(state.index, {
        includeScore: true,
        threshold: 0.35,
        distance: 200,
        ignoreLocation: true,
        keys: [
          { name: "title", weight: 0.6 },
          { name: "summary", weight: 0.2 },
          { name: "content", weight: 0.2 },
          { name: "tags", weight: 0.1 },
          { name: "categories", weight: 0.1 },
        ],
      });
      return state.fuse;
    });
  };

  const buildResultMarkup = (item) => {
    const article = document.createElement("article");
    article.className = "post-item search-result";

    const topRow = document.createElement("div");
    topRow.className = "search-result__top";

    const titleEl = document.createElement("h4");
    titleEl.className = "post-item-title";
    const link = document.createElement("a");
    link.href = item.permalink;
    link.textContent = item.title;
    titleEl.appendChild(link);

    const metaEl = document.createElement("div");
    metaEl.className = "post-item-right search-result__meta";
    const categories = Array.isArray(item.categories) ? item.categories.filter(Boolean) : [];
    if (categories.length) {
      const categorySpan = document.createElement("span");
      // categorySpan.className = "search-result__category";
      categorySpan.className = "post-item-meta";
      categorySpan.textContent = categories.join(", ");
      metaEl.appendChild(categorySpan);
    } else if (item.section) {
      const sectionSpan = document.createElement("span");
      // sectionSpan.className = "search-result__category";
      sectionSpan.className = "post-item-meta";
      sectionSpan.textContent = item.section;
      metaEl.appendChild(sectionSpan);
    }
    if (item.date) {
      if (metaEl.childNodes.length) {
        const divider = document.createElement("span");
        divider.textContent = "|";
        metaEl.appendChild(divider);
      }
      const timeEl = document.createElement("time");
      timeEl.className = "post-item-meta";
      timeEl.textContent = item.date;
      metaEl.appendChild(timeEl);
    }

    topRow.appendChild(titleEl);
    topRow.appendChild(metaEl);
    article.appendChild(topRow);

    const summary = item.summary || item.content || "";
    if (summary) {
      const excerptEl = document.createElement("p");
      // excerptEl.className = "search-result__excerpt";
      excerptEl.className = "post-item-meta";
      excerptEl.textContent = summary.length > 160 ? `${summary.slice(0, 157)}…` : summary;
      article.appendChild(excerptEl);
    }

    return article;
  };

  const renderResults = (items, query) => {
    resultsContainer.innerHTML = "";
    if (!items.length) {
      messageContainer.textContent = `“${query}”에 대한 검색 결과가 없습니다.`;
      return;
    }

    messageContainer.textContent = `“${query}” 검색 결과: ${items.length}건`;

    const fragment = document.createDocumentFragment();
    items.forEach((entry) => {
      fragment.appendChild(buildResultMarkup(entry.item));
    });

    resultsContainer.appendChild(fragment);
  };

  const clearResults = (message) => {
    resultsContainer.innerHTML = "";
    messageContainer.textContent = message || "";
  };

  const params = new URLSearchParams(window.location.search);
  const initialQuery = params.get("q") || "";

  const query = initialQuery.trim();
  if (!query) {
    clearResults("검색어가 비어 있습니다. 상단 검색창에서 키워드를 입력해 주세요.");
    return;
  }

  if (query.length < MIN_QUERY_LENGTH) {
    clearResults("검색어를 두 글자 이상 입력해 주세요.");
    return;
  }

  clearResults(`“${query}” 검색 중입니다…`);

  ensureFuse()
    .then((fuse) => fuse.search(query))
    .then((results) => renderResults(results, query))
    .catch((error) => {
      console.error(error);
      clearResults("검색 중 오류가 발생했습니다.");
    });
})();
