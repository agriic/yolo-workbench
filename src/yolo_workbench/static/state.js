export const state = {
  mode: "single",      // "single" dataset or "projects" folder
  server: false,       // started by yolo-workbench-server
  features: { path_import: true, model_upload: true },  // from /api/v1/app
  project: null,       // project slug in projects mode
  user: null,          // the signed-in user {name, display_name, admin}, or null without sign-in
  meta: null,
  revision: 0,         // the dataset revision this page has seen (collab.js polls for later changes)
  feed: null,          // which opening of the dataset `revision` counts in (a reopened project starts again)
  presence: {},        // image id -> other people who have it open in their editor
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
  focusHint: null,     // points of a shape outlined in the editor (a Label QA prediction), or null
  drawing: null,       // {start,end} box or {points,cursor} polygon in progress
  drag: null,          // {mode:"move"|"resize"|"vertex", a, index, start, original}
  polyHover: null,     // selected polygon under the mouse: {a, vertex, remove} or {a, edge, point} (where a press adds a vertex)
  pan: null,
  spaceDown: false,
  view: { scale: 1, x: 0, y: 0 },
  canUndo: false,
  canRedo: false,
  pred: {
    status: "unavailable", error: null, names: {}, mapping: {}, pending: {},
    modelPath: null, unloaded: null, job: { state: "idle", done: 0, total: 0 }, items: [], minConf: 0, maxPoints: null, poll: null, models: [],
    hovered: null, hoverSource: null,  // the prediction under the mouse, and where: "list" (its row) or "canvas" (its ghost)
  },
  sam: {
    status: "unavailable", model: null, message: null, error: null, unloaded: null, poll: null,
    active: false,       // smart annotation mode is on
    points: [],          // {x, y, positive} prompts, normalized
    box: null,           // [x1, y1, x2, y2] box prompt, normalized
    press: null, drag: null,
    result: null,        // {found, score, mask, points} from the server: SAM's mask outline, what ⏎ adds
    busy: false, request: 0, encoding: null, encoded: null,
  },
  embed: {
    status: "idle", error: null, items: [], hovered: null, poll: null,
    classes: new Set(), selection: [], band: null, mode: "3d",
    rotation: { yaw: -0.65, pitch: -0.35 }, zoom: 1, rotate: null, projected: [],
    pan2d: { x: 0, y: 0 }, zoom2d: 1, panDrag: null,
  },
};

