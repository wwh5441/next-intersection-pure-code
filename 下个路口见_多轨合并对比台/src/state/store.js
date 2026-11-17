export class VocalStudioStore {
  constructor(initialCombo, initialProfile, initialTime = 0) {
    this.state = {
      combo: initialCombo,
      profile: initialProfile,
      time: initialTime,
      running: false,
      loading: false,
      loadProgress: null,
      volume: 0.9,
      solo: null,
      activePreset: 'full_mix',
      computed: 0,
      meters: {},
      enabled: {},
      gains: {},
    };
    this.listeners = new Set();
  }

  initTracks(tracks) {
    const enabled = {};
    const gains = {};
    for (const track of tracks) {
      gains[track.id] = 1;
      enabled[track.id] = true;
    }
    this.state.gains = gains;
    this.state.enabled = enabled;
    this.state.solo = null;
    this.state.activePreset = 'full_mix';
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
