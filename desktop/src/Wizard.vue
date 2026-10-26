<script setup lang="ts">
import { computed, reactive, ref } from "vue";
import { balance, openLink as openUrl, saveSettings, usd, type Hardware, type Kind, type Prefs } from "./api";

const props = defineProps<{ prefs: Prefs; hardware: Hardware | null; hasKey: boolean }>();
const emit = defineEmits<{ done: [kind: Kind] }>();

const RUNPOD_SIGNUP = "https://runpod.io?ref=cazj8c05";

const kind = ref<Kind | null>(null);
const step = ref(0);
const prefs = reactive<Prefs>({ ...props.prefs });
const secrets = reactive({ runpodKey: "" });
const checked = ref<number | null>(null);
const checking = ref(false);
const problem = ref("");
const saving = ref(false);

const steps = computed(() =>
  kind.value === "local" ? ["Where", "What happens next"] : ["Where", "RunPod account", "Storage", "Safety", "What happens next"],
);
const last = computed(() => step.value === steps.value.length - 1);
const canNext = computed(() => {
  if (step.value === 0) return kind.value !== null;
  if (kind.value === "runpod" && step.value === 1) return checked.value !== null || (props.hasKey && !secrets.runpodKey);
  return true;
});

function choose(where: Kind) {
  if (where === "local" && !props.hardware?.ok) return;
  kind.value = where;
}

async function checkKey() {
  checking.value = true;
  problem.value = "";
  try {
    checked.value = await balance(secrets.runpodKey || undefined);
  } catch (e) {
    checked.value = null;
    problem.value = String(e);
  } finally {
    checking.value = false;
  }
}

async function next() {
  if (!last.value) {
    step.value += 1;
    return;
  }
  saving.value = true;
  problem.value = "";
  try {
    await saveSettings({ ...prefs, setup_done: true }, secrets);
    emit("done", kind.value!);
  } catch (e) {
    problem.value = String(e);
  } finally {
    saving.value = false;
  }
}
</script>

<template>
  <div class="wizard">
    <ol class="trail">
      <li v-for="(name, i) in steps" :key="name" :class="{ on: i === step, done: i < step }">
        <span class="mono">{{ String(i + 1).padStart(2, "0") }}</span>{{ name }}
      </li>
    </ol>

    <section v-if="step === 0" class="stack">
      <div>
        <h2>Where should DiffLab run?</h2>
        <p class="muted">It needs an NVIDIA card with 24 GB of memory and 64 GB of RAM. If this PC has that, it runs here for free. If not, it rents one on RunPod by the second.</p>
      </div>
      <button class="option" :class="{ on: kind === 'local' }" :disabled="!hardware?.ok" @click="choose('local')">
        <span class="eyebrow">This PC</span>
        <strong>Run on my own GPU</strong>
        <span v-if="!hardware" class="muted">Checking this PC…</span>
        <span v-else-if="hardware.ok" class="muted">{{ hardware.gpu }} · {{ hardware.ram_gb.toFixed(0) }} GB RAM · {{ hardware.runtime }}</span>
        <span v-else class="muted">Not possible here: {{ hardware.reason }}.</span>
      </button>
      <button class="option" :class="{ on: kind === 'runpod' }" @click="choose('runpod')">
        <span class="eyebrow">RunPod cloud</span>
        <strong>Rent a GPU when I need it</strong>
        <span class="muted">An RTX 3090, 4090 or 5090 (Settings lists other GPUs with their prices), billed by the second from your RunPod balance, and stopped for you when you are done.</span>
      </button>
    </section>

    <section v-else-if="kind === 'runpod' && step === 1" class="stack">
      <div>
        <h2>Connect your RunPod account</h2>
        <p class="muted">The GPU is rented on your own account, so you pay RunPod directly and see every charge there.</p>
      </div>
      <ol class="how">
        <li>Create an account at <a href="#" @click.prevent="openUrl(RUNPOD_SIGNUP)">runpod.io</a>.</li>
        <li>Add credit under Billing. $10 is enough to try it.</li>
        <li>Under Credentials → API Keys, create a key with <em>All</em> access and paste it here.</li>
      </ol>
      <label class="field">
        <span class="label">API key</span>
        <div class="row">
          <input v-model="secrets.runpodKey" type="password" :placeholder="hasKey ? 'saved; paste a new one to replace it' : 'rpa_…'" @input="checked = null" />
          <button :disabled="checking || (!secrets.runpodKey && !hasKey)" @click="checkKey">{{ checking ? "Checking…" : "Check" }}</button>
        </div>
      </label>
      <p v-if="checked !== null" class="band delivered">The key works. Your balance is {{ usd(checked) }}.</p>
      <p v-if="problem" class="band error">{{ problem }}</p>
    </section>

    <section v-else-if="kind === 'runpod' && step === 2" class="stack">
      <div>
        <h2>Where your work is kept</h2>
        <p class="muted">Projects and finished videos live on a RunPod <strong>global volume</strong> named difflab, which the app creates the first time. Any GPU in any region can use it, so a start does not wait for one datacenter to have a free card.</p>
      </div>
      <ul class="how">
        <li>It grows as you use it, and you pay RunPod for what it holds and for reads and writes. The first GPU also keeps a copy of the models there (about 96 GB); every later one copies it onto its own disk rather than downloading from Hugging Face.</li>
        <li>It stays when no GPU is running. Delete it in RunPod's Storage page if you stop using DiffLab.</li>
        <li>Finished videos download to this PC while a GPU is running.</li>
      </ul>
    </section>

    <section v-else-if="kind === 'runpod' && step === 3" class="stack">
      <div>
        <h2>So a GPU is never left running</h2>
        <p class="muted">The rented machine watches itself and deletes itself when any of these happens, even if this app is closed or your computer is asleep. Your work stays on the storage.</p>
      </div>
      <div class="knob">
        <div><span class="name">Nothing to do</span><span class="hint">Minutes with an empty queue and this app closed.</span></div>
        <input v-model.number="prefs.idle_minutes" type="number" min="2" max="30" />
      </div>
      <div class="knob">
        <div><span class="name">Spend limit</span><span class="hint">Dollars per session. It stops even in the middle of a render.</span></div>
        <input v-model.number="prefs.spend_limit_usd" type="number" min="1" step="1" />
      </div>
      <div class="knob">
        <div><span class="name">Stuck</span><span class="hint">Minutes a render may go without progress.</span></div>
        <input v-model.number="prefs.stuck_minutes" type="number" min="5" max="60" />
      </div>
      <label class="check"><input v-model="prefs.exit_when_done" type="checkbox" /> Stop as soon as the queue is done, even with the app open</label>
    </section>

    <section v-else class="stack">
      <div>
        <h2>What happens next</h2>
      </div>
      <ol v-if="kind === 'runpod'" class="how">
        <li><strong>Run on RunPod</strong> rents a GPU. The first time it also creates your storage; the machine takes a few minutes to boot.</li>
        <li><strong>Open DiffLab</strong> opens the studio in its own window, as if it ran on this PC.</li>
        <li>Each session's <strong>first render</strong> of a kind downloads its models (about 96 GB in all) onto the GPU's own disk, a few minutes at most.</li>
        <li>The cost so far and your balance stay on screen. Close the app whenever you like: the GPU deletes itself after {{ prefs.idle_minutes }} idle minutes.</li>
        <li><strong>Results</strong> copies finished videos to this PC.</li>
      </ol>
      <ol v-else class="how">
        <li><strong>Run on this PC</strong> downloads the DiffLab image (3.6 GB, once) and starts it in Docker.</li>
        <li><strong>Open DiffLab</strong> opens the studio in its own window.</li>
        <li>The <strong>first render</strong> downloads about 96 GB of models into Docker's <span class="mono">difflab</span> volume, so it is slow.</li>
        <li>It uses the whole GPU while it runs. Press <strong>Stop</strong> when you are done.</li>
      </ol>
      <p v-if="problem" class="band error">{{ problem }}</p>
    </section>

    <footer class="foot">
      <button class="quiet" :disabled="step === 0" @click="step -= 1">Back</button>
      <button class="primary" :disabled="!canNext || saving" @click="next">{{ last ? "Finish" : "Continue" }}</button>
    </footer>
  </div>
</template>

<style scoped>
.wizard {
  display: flex;
  flex-direction: column;
  gap: var(--space-3);
  min-height: 100%;
}
.trail {
  list-style: none;
  margin: 0;
  padding: 0;
  display: flex;
  flex-wrap: wrap;
  gap: var(--space-1) var(--space-2);
  font-family: var(--font-mono);
  font-size: 11px;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--text-tertiary);
}
.trail li span {
  margin-right: 6px;
}
.trail .on {
  color: var(--accent);
}
.trail .done {
  color: var(--text-secondary);
}
h2 {
  font-size: var(--text-xl);
  margin: 0 0 var(--space-1);
}
h3 {
  font-size: var(--text-md);
  margin: 0 0 4px;
}
p {
  margin: 0;
}
.option {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  gap: 4px;
  text-align: left;
  padding: var(--space-2);
  background: var(--surface);
  border-color: var(--border);
  font-size: var(--text-base);
  font-weight: var(--weight-regular);
}
.option strong {
  font-family: var(--font-heading);
  font-size: var(--text-md);
  font-weight: var(--weight-semibold);
  color: var(--text-primary);
}
.option.on {
  border-color: var(--accent);
  background: var(--bg-raised);
}
.option:disabled strong {
  color: var(--dim);
}
.how {
  margin: 0;
  padding-left: 20px;
  display: flex;
  flex-direction: column;
  gap: var(--space-1);
  color: var(--text-secondary);
}
.how strong {
  color: var(--text-primary);
  font-weight: var(--weight-medium);
}
.field {
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.row {
  display: flex;
  gap: var(--space-1);
}
.check {
  display: flex;
  gap: var(--space-1);
  align-items: center;
  color: var(--text-secondary);
}
.foot {
  margin-top: auto;
}
</style>
