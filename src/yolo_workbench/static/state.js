export const state = {
  mode: "single",      // "single" dataset or "projects" folder
  project: null,       // project slug in projects mode
  meta: null,
  imageSelection: new Map(),  // image id -> image item (Annotation grid bulk actions)
  lastImageIndex: null,
  imageSelectionBusy: false,
  objectSelection: new Map(),  // "imageId|annotationId" -> object item
  lastObjectIndex: null,
  objectSelectionBusy: false,
  editorGridIndex: null,
  editorDetached: false,  // editor image dropped out of the filtered grid; editorGridIndex is where it was
  issues: [],
  issueFilter: null,
  statistics: null,
  detail: null,        // image detail currently in the editor
  img: null,           // loaded HTMLImageElement
  selected: null,      // annotation id
  hovered: null,
  drawing: null,       // {start,end} box or {points,cursor} polygon in progress
  drag: null,          // {mode:"move"|"resize"|"vertex", a, index, start, original}
  pan: null,
  spaceDown: false,
  view: { scale: 1, x: 0, y: 0 },
  canUndo: false,
  canRedo: false,
  pred: {
    status: "unavailable", error: null, names: {}, mapping: {}, pending: {},
    job: { state: "idle", done: 0, total: 0 }, items: [], minConf: 0, poll: null, models: [],
  },
  sam: {
    status: "unavailable", model: null, message: null, error: null, poll: null,
    active: false,       // smart annotation mode is on
    points: [],          // {x, y, positive} prompts, normalized
    box: null,           // [x1, y1, x2, y2] box prompt, normalized
    press: null, drag: null,
    result: null,        // {found, score, outline, points} from the server
    busy: false, request: 0, encoding: null, encoded: null,
  },
  embed: {
    status: "idle", error: null, items: [], hovered: null, poll: null,
    classes: new Set(), selection: [], band: null, mode: "3d",
    rotation: { yaw: -0.65, pitch: -0.35 }, zoom: 1, rotate: null, projected: [],
    pan2d: { x: 0, y: 0 }, zoom2d: 1, panDrag: null,
  },
};

