const addForm = document.getElementById('add-form');
const todoInput = document.getElementById('todo-input');
const todoList = document.getElementById('todo-list');
const loadingEl = document.getElementById('loading');
const errorBanner = document.getElementById('error-banner');

let inflight = 0;

function busy(on) {
    inflight += on ? 1 : -1;
    loadingEl.classList.toggle('hidden', inflight === 0);
}

async function api(method, path, body) {
    busy(true);
    errorBanner.classList.add('hidden');

    try {
        const opts = { method };
        if (body !== undefined) {
            opts.headers = { 'Content-Type': 'application/json' };
            opts.body = JSON.stringify(body);
        }
        const res = await fetch(path, opts);
        if (res.status === 204) return null;
        const data = await res.json();
        if (!res.ok) {
            showError(data.error || 'Unknown error');
            return null;
        }
        return data;
    } catch (err) {
        showError('Network error: ' + err.message);
        return null;
    } finally {
        busy(false);
    }
}

function showError(msg) {
    errorBanner.textContent = msg;
    errorBanner.classList.remove('hidden');
    setTimeout(() => errorBanner.classList.add('hidden'), 5000);
}

function clearChildren(el) {
    while (el.firstChild) el.removeChild(el.firstChild);
}

function renderTodos(todos) {
    clearChildren(todoList);

    if (!todos || todos.length === 0) {
        const li = document.createElement('li');
        li.className = 'empty';
        li.textContent = 'No todos yet. Add one above.';
        todoList.appendChild(li);
        return;
    }

    todos.forEach(todo => {
        const li = document.createElement('li');
        if (todo.completed) li.className = 'completed';

        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.checked = todo.completed;
        checkbox.disabled = todo.completed;
        checkbox.addEventListener('change', () => handleComplete(todo.id));

        const span = document.createElement('span');
        span.textContent = todo.title;

        if (todo.priority || todo.time) {
            const meta = document.createElement('small');
            const parts = [];
            if (todo.priority) parts.push(todo.priority);
            if (todo.time) parts.push(todo.time);
            meta.textContent = ' (' + parts.join(', ') + ')';
            meta.className = 'meta';
            span.appendChild(meta);
        }

        const deleteBtn = document.createElement('button');
        deleteBtn.textContent = 'Delete';
        deleteBtn.className = 'delete-btn';
        deleteBtn.addEventListener('click', () => handleDelete(todo.id));

        li.append(checkbox, span, deleteBtn);
        todoList.appendChild(li);
    });
}

async function loadTodos() {
    const todos = await api('GET', '/api/todos');
    if (todos) renderTodos(todos);
}

addForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const title = todoInput.value.trim();
    if (!title) return;

    await api('POST', '/api/todos', { title });
    todoInput.value = '';
    await loadTodos();
});

async function handleComplete(id) {
    await api('PATCH', '/api/todos/' + id);
    await loadTodos();
}

async function handleDelete(id) {
    await api('DELETE', '/api/todos/' + id);
    await loadTodos();
}

document.addEventListener('DOMContentLoaded', loadTodos);
