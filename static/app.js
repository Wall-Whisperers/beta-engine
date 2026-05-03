const typeColors = {
  jug: '#22c55e',
  crimp: '#ef4444',
  sloper: '#f59e0b',
  pinch: '#3b82f6',
  foothold: '#a855f7',
};

const state = {
  rows: 10,
  cols: 8,
  selected: null,
  holds: {},
};

const wall = document.getElementById('wall');
const selection = document.getElementById('selection');
const legend = document.getElementById('legend');

const rowsInput = document.getElementById('rows');
const colsInput = document.getElementById('cols');
const holdType = document.getElementById('holdType');
const orientation = document.getElementById('orientation');
const size = document.getElementById('size');
const holdColor = document.getElementById('holdColor');
const isStart = document.getElementById('isStart');
const isFinish = document.getElementById('isFinish');
const output = document.getElementById('jsonOutput');

function holdKey(row, col) {
  return `${row},${col}`;
}

function renderLegend() {
  legend.innerHTML = '';
  Object.entries(typeColors).forEach(([type, color]) => {
    const chip = document.createElement('span');
    chip.className = 'chip';
    chip.innerHTML = `<span class="dot" style="background:${color}"></span>${type}`;
    legend.appendChild(chip);
  });
}

function renderWall() {
  wall.style.gridTemplateColumns = `repeat(${state.cols}, 42px)`;
  wall.innerHTML = '';

  for (let r = 0; r < state.rows; r += 1) {
    for (let c = 0; c < state.cols; c += 1) {
      const cell = document.createElement('button');
      cell.className = 'cell';
      if (state.selected && state.selected.row === r && state.selected.col === c) {
        cell.classList.add('selected');
      }
      const key = holdKey(r, c);
      const hold = state.holds[key];
      if (hold) {
        const marker = document.createElement('div');
        marker.className = 'marker';
        marker.style.background = hold.color || typeColors[hold.type] || '#999';
        marker.style.transform = `rotate(${{ up: 0, right: 90, down: 180, left: 270 }[hold.orientation]}deg)`;
        marker.style.opacity = hold.size === 's' ? 0.75 : hold.size === 'm' ? 0.9 : 1;
        marker.textContent = `${hold.isStart ? 'S' : ''}${hold.isFinish ? 'F' : ''}` || hold.type[0].toUpperCase();
        cell.appendChild(marker);
      }
      cell.addEventListener('click', () => selectCell(r, c));
      wall.appendChild(cell);
    }
  }
}

function selectCell(row, col) {
  state.selected = { row, col };
  const hold = state.holds[holdKey(row, col)];
  selection.textContent = `Row ${row}, Col ${col}`;
  if (hold) {
    holdType.value = hold.type;
    orientation.value = hold.orientation;
    size.value = hold.size;
    holdColor.value = hold.color;
    isStart.checked = !!hold.isStart;
    isFinish.checked = !!hold.isFinish;
  } else {
    holdType.value = 'jug';
    orientation.value = 'up';
    size.value = 'm';
    holdColor.value = typeColors.jug;
    isStart.checked = false;
    isFinish.checked = false;
  }
  renderWall();
}

function currentSchema() {
  return {
    schema_version: '1.0',
    grid: { rows: state.rows, cols: state.cols },
    holds: Object.values(state.holds),
  };
}

document.getElementById('resize').addEventListener('click', () => {
  state.rows = Math.max(1, Number(rowsInput.value));
  state.cols = Math.max(1, Number(colsInput.value));
  state.selected = null;

  Object.keys(state.holds).forEach((key) => {
    const [row, col] = key.split(',').map(Number);
    if (row >= state.rows || col >= state.cols) delete state.holds[key];
  });

  selection.textContent = 'None';
  renderWall();
});

document.getElementById('saveHold').addEventListener('click', () => {
  if (!state.selected) return;
  const { row, col } = state.selected;
  const hold = {
    row,
    col,
    type: holdType.value,
    orientation: orientation.value,
    size: size.value,
    color: holdColor.value,
    isStart: isStart.checked,
    isFinish: isFinish.checked,
  };
  if (!hold.color) hold.color = typeColors[hold.type];
  state.holds[holdKey(row, col)] = hold;
  renderWall();
});

document.getElementById('removeHold').addEventListener('click', () => {
  if (!state.selected) return;
  delete state.holds[holdKey(state.selected.row, state.selected.col)];
  renderWall();
});

document.getElementById('exportJson').addEventListener('click', () => {
  output.value = JSON.stringify(currentSchema(), null, 2);
});

holdType.addEventListener('change', () => {
  if (!state.selected) return;
  const key = holdKey(state.selected.row, state.selected.col);
  const hold = state.holds[key];
  if (!hold) holdColor.value = typeColors[holdType.value];
});

renderLegend();
renderWall();
output.value = JSON.stringify(currentSchema(), null, 2);
