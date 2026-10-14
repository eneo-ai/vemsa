# Vemsa quick start

Vemsa identifies speakers. It takes a recording and a transcript that another model wrote,
and returns the transcript with a label such as `SPEAKER_00` on each line. Connected to Eneo
it never writes the text: the flow's transcription model does, for example
[vadsa](https://github.com/eneo-ai/vadsa). This guide sets up only what Eneo uses.

## What you need

- A `linux/amd64` host with Docker Compose, which also starts the PostgreSQL database that
  Vemsa needs. There is no arm64 image.
- An NVIDIA GPU with its driver and `nvidia-container-toolkit` on the host. Production needs
  the GPU. A host without one can try Vemsa with the CPU image, see below.
- A Hugging Face token for an account that has accepted the license of the gated model
  `pyannote/speaker-diarization-community-1`.

## Start it

You must set three values: the database password, the token Eneo will use, and the Hugging
Face token. Put them in a file named `.env` next to `compose.yaml`; Compose reads it on every
start. A token has the form `client_id=token`, and Eneo's key is the part after `=`.
`VEMSA_ENGINE=diarize` loads no speech recognition model, because Eneo sends the text.

```
POSTGRES_PASSWORD=change-me-long-random-value
VEMSA_API_TOKENS=eneo=change-me-another-long-random-value
HF_TOKEN=hf_change-me
VEMSA_ENGINE=diarize
```

```bash
git clone https://github.com/eneo-ai/vemsa.git
cd vemsa
# create .env with the four lines above, then:
docker compose pull api worker      # fetch the published image
docker compose up -d --no-build     # start it without building from source
```

This publishes the API on `127.0.0.1:8000`. To let Eneo's containers on the same host reach
it by name instead, start it on a shared Docker network. When anything but your own host can
reach the API, which includes every container on that network, also add `VEMSA_OPS_USER` and
`VEMSA_OPS_PASSWORD` to `.env`: the operator dashboard at `/ops` is open without them.

```bash
docker network create eneo-shared    # once, or reuse the network Eneo already uses
docker compose -f compose.yaml -f compose.eneo.yaml up -d --no-build
```

To try Vemsa on a host without a GPU, replace `docker compose pull` and `docker compose up`
with these commands. Vemsa is slow on a CPU, and the override needs Compose 2.24 or later.

```bash
printf 'services:\n  worker:\n    deploy: !reset {}\n' > compose.cpu.yaml
VEMSA_IMAGE=ghcr.io/eneo-ai/vemsa:latest-cpu \
  docker compose -f compose.yaml -f compose.cpu.yaml up -d --no-build
```

## Check that it runs

```bash
curl -s http://127.0.0.1:8000/v1/health/ready \
  -H "Authorization: Bearer change-me-another-long-random-value"
```

A good answer has status `200` and contains `"status":"ready"`, `"queue_accepting_jobs":true`
and `"supported_tasks":["align","diarize","timestamp"]`. `503` with `"status":"not_ready"`
means the database or the worker is not up yet. `401` means the token is wrong.

## Connect it to Eneo

Eneo's backend and flow execution worker must reach Vemsa. As an administrator in Eneo, enter:

| Setup | Address |
| --- | --- |
| Shared Docker network (`compose.eneo.yaml`) | `http://vemsa:8000` |
| Published port | `http://<VEMSA_BIND>:<VEMSA_PORT>`. The defaults give `http://127.0.0.1:8000`, which only the host itself reaches, not other containers, so set `VEMSA_BIND` to a private address. |

1. Open **Admin → Models → Transcription models**, section **Speaker identification**
   (**Administratör → Modeller → Transkription**, **Talaridentifiering**).
2. Choose **Connect service** (**Anslut tjänst**).
3. Enter a **Name** (**Namn**), the **Address** (**Adress**) from the table, and as
   **API key** (**API-nyckel**) the part after `eneo=` in `VEMSA_API_TOKENS`.
4. Choose **Connect** (**Anslut**). Check that the status reads **Accepting jobs**
   (**Tar emot jobb**) with "The service can identify speakers."
5. Open a space's **Settings** (**Inställningar**). Under **Advanced Settings**
   (**Avancerade inställningar**), switch the service on in **Speaker identification**.
6. In the **Transcription** stage (**Transkribering**) of a flow that transcribes audio,
   turn on **Speaker identification** and publish the flow.
7. Check the result. Run the flow with a recording of two people. Each line of the
   transcript starts like `[00:00:12 - 00:00:15] SPEAKER_00:`.

## Use it

A person who runs the flow gets the transcript that the flow's transcription model wrote,
with `SPEAKER_00`, `SPEAKER_01` and so on in front of each line. Vemsa decides where the
lines break and which label each line gets.

## Next

- [Production operations](PRODUCTION.md): image tags, the dashboard, and the Eneo checklist
  under [Consumer integrations](PRODUCTION.md#consumer-integrations-eneo).
- [Engine tiers](../README.md#engine-tiers) and [Configuration](../README.md#configuration):
  what Vemsa does on its own, and every setting.
- [Architecture](ARCHITECTURE.md#eneo-and-vemsa-the-responsibility-split): who owns what.
- The [vadsa quick start](https://github.com/eneo-ai/vadsa/blob/main/docs/quickstart.md): a
  transcription server for Eneo.
