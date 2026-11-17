export class VocalStudioStore {
  constructor(initialProfile, initialTime = 22.6) {
    this.state = {
      profile: initialProfile,
      time: initialTime,
      running: false,
      loading: false,
      loadProgress: null,
      volume: 0.9,
      solo: null,
      activePreset: 'all_vocal',
      computed: 0,
      meters: {},
      enabled: {},
      gains: {},
    };
    this.listeners = new Set();
  }

  initTracks(tracks) {
    for (const track of tracks) {
      if (!(track.id in this.state.gains)) this.state.gains[track.id] = 1;
      if (!(track.id in this.state.enabled)) this.state.enabled[track.id] = true;
    }
    this.emit();
  }

  patch(partial) {
    Object.assign(this.state, partial);
    this.emit();
  }

  subscribe(fn) {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  }

  emit() {
    for (const fn of this.listeners) fn(this.state);
  }
}
