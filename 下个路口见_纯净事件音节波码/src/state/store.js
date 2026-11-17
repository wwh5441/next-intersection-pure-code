export class StudioStore {
  constructor(initialState) {
    this.state = {
      profile: 'expressive',
      loading: true,
      loadingText: '正在初始化事件音节生波引擎…',
      playing: false,
      currentTime: 0,
      duration: 210.373,
      synthMs: 0,
      activeEventIndex: -1,
      highlightedSyllableId: null,
      layerStates: {},
      ...initialState,
    };
    this.listeners = new Set();
  }

  getState() {
    return this.state;
  }

  setState(patch) {
    Object.assign(this.state, patch);
    for (const fn of this.listeners) {
      fn(this.state);
    }
  }

  subscribe(fn) {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  }
}
