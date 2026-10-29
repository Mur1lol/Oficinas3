import json
import time

import serial
import serial.tools.list_ports


class ESP32Communication:
    """Comunicação USB/serial (JSON por linha) entre o computador/Raspberry e o ESP32.

    O ESP32 controla os motores do Gantry (CoreXY), o eletroímã e os fins de curso:

    gantry    -> {"comando": "gantry",
                  "resposta": {"x": [...], "y": [...], "eletroima": [0|1, ...]}}
                 As 3 listas têm o mesmo tamanho: o passo i move o gantry para
                 (x[i], y[i]) mm e define o eletroímã (1 = ligado, 0 = desligado).
                 O ESP32 responde {"comando": "ok"} ao TERMINAR toda a trajetória.

    homing    -> {"comando": "homing"}
                 Executa o ciclo de homing procurando os fins de curso X e Y.
                 O ESP32 responde {"comando": "ok"} ao concluir.

    eletroima -> {"comando": "eletroima", "resposta": 1|0}
                 Liga ou desliga o eletroímã diretamente.

    status    -> {"comando": "status"}
                 Retorna o estado atual: posição X/Y, homed, drivers, eletroímã e fins de curso.
    """

    GANTRY_TIMEOUT = 120.0   # s, tempo máximo aguardando o gantry terminar movimentos
    COMANDO_TIMEOUT = 15.0   # s, tempo máximo para comandos rápidos

    def __init__(self, porta=None, baudrate=115200, timeout=1, verbose=True):
        self.porta = porta
        self.baudrate = baudrate
        self.timeout = timeout
        self.verbose = verbose
        self.ser = None

    # ------------------------------------------------------------------
    # Conexão
    # ------------------------------------------------------------------

    def _log(self, msg):
        if self.verbose:
            print(msg)

    @staticmethod
    def encontrar_porta_esp32(verbose=True):
        """Procura automaticamente a porta do ESP32 (ou a primeira disponível)."""
        ports = serial.tools.list_ports.comports()
        for port in ports:
            # Procura por dispositivos USB que podem ser ESP32 (Windows e Linux/Raspberry)
            info = f"{port.device} {port.description} {port.hwid}".lower()
            if any(k in info for k in ('ch340', 'cp210', 'esp32', 'uart', 'ttyusb', 'ttyacm', 'usb')):
                if verbose:
                    print(f"Porta ESP32 encontrada: {port.device} - {port.description}")
                return port.device

        if verbose:
            print("Portas disponíveis:")
            for port in ports:
                print(f"  {port.device} - {port.description}")
        return ports[0].device if ports else None

    def conectar(self):
        """Abre a porta serial. Levanta ConnectionError se falhar."""
        if self.is_connected:
            return
        porta = self.porta or self.encontrar_porta_esp32(self.verbose)
        if porta is None:
            raise ConnectionError("Nenhuma porta serial encontrada para o ESP32.")
        try:
            self.ser = serial.Serial(porta, self.baudrate, timeout=self.timeout)
        except Exception as e:
            raise ConnectionError(f"Erro ao conectar à porta {porta}: {e}") from e
        self.porta = porta
        self._log(f"Conectado à porta {porta}")
        time.sleep(2)  # o ESP32 reinicia ao abrir a serial
        self.ser.reset_input_buffer()

    def fechar(self):
        """Fecha a conexão serial."""
        if self.is_connected:
            self.ser.close()
            self._log("Conexão serial fechada.")

    close = fechar

    @property
    def is_connected(self):
        return self.ser is not None and self.ser.is_open

    def __enter__(self):
        self.conectar()
        return self

    def __exit__(self, *_):
        self.fechar()

    # ------------------------------------------------------------------
    # Mensagens
    # ------------------------------------------------------------------

    def enviar_comando(self, cmd, rsp=None):
        """Envia um comando JSON para o ESP32 via USB."""
        payload = {"comando": cmd}
        if rsp is not None:
            payload["resposta"] = rsp
        comando_json = json.dumps(payload) + '\n'
        self.ser.write(comando_json.encode())
        self._log(f"Enviado: {comando_json.strip()}")

    def receber_mensagem(self, timeout=None):
        """Recebe uma linha JSON do ESP32 (ignora linhas de log)."""
        inicio = time.monotonic()
        while True:
            raw = self.ser.readline()
            linha = raw.decode('utf-8', errors='ignore').strip()
            if linha:
                try:
                    dado = json.loads(linha)
                    self._log(f"Recebido: {dado}")
                    return dado
                except json.JSONDecodeError:
                    self._log(f"Ignorado texto não JSON: {linha}")
            if timeout is not None and time.monotonic() - inicio > timeout:
                return None
            time.sleep(0.05)

    def _aguardar_ok(self, timeout, descricao):
        msg = self.receber_mensagem(timeout=timeout)
        if msg is None:
            raise TimeoutError(f"ESP32 não respondeu a '{descricao}' em {timeout:.0f}s.")
        if msg.get("comando") != "ok":
            motivo = msg.get("motivo", str(msg))
            raise RuntimeError(f"Erro do ESP32 em '{descricao}': {motivo}")
        return msg

    # ------------------------------------------------------------------
    # Comandos do protocolo
    # ------------------------------------------------------------------

    def gantry(self, x, y, eletroima, timeout=None):
        """Executa uma trajetória no gantry.

        x, y      : posições em X e Y (em mm) para cada passo
        eletroima : 1/True = ligado, 0/False = desligado em cada passo
        As 3 listas devem ter exatamente o mesmo tamanho.
        Bloqueia até o ESP32 responder "ok" ao concluir todos os passos.
        """
        if not (len(x) == len(y) == len(eletroima)):
            raise ValueError(
                f"x, y e eletroima devem ter o mesmo tamanho "
                f"({len(x)}, {len(y)}, {len(eletroima)})."
            )
        self.ser.reset_input_buffer()
        self.enviar_comando("gantry", {
            "x": [float(v) for v in x],
            "y": [float(v) for v in y],
            "eletroima": [1 if e else 0 for e in eletroima],
        })
        return self._aguardar_ok(timeout or self.GANTRY_TIMEOUT, "gantry")

    def homing(self, timeout=120.0):
        # Pior caso: ~450 mm em X + ~850 mm em Y a 15 mm/s = ~90 s
        """Solicita o homing do gantry nos eixos X e Y."""
        self.ser.reset_input_buffer()
        self.enviar_comando("homing")
        return self._aguardar_ok(timeout, "homing")

    def mover_eixo(self, eixo: str, distancia_mm: float, timeout=30.0):
        """Envia comando para o ESP32 mover o gantry em um eixo específico pela distância em mm.

        eixo: 'X' ou 'Y'
        distancia_mm: distância em mm (positivo ou negativo)
        """
        eixo_str = str(eixo).strip().upper()
        if eixo_str not in ("X", "Y"):
            raise ValueError(f"Eixo inválido: '{eixo}'. Deve ser 'X' ou 'Y'.")
        self.ser.reset_input_buffer()
        self.enviar_comando("mover", {
            "eixo": eixo_str,
            "distancia": float(distancia_mm),
        })
        return self._aguardar_ok(timeout, f"mover {eixo_str} {distancia_mm}mm")

    def acionar_eletroima(self, ligar=True, timeout=None):
        """Liga ou desliga o eletroímã imediatamente."""
        self.ser.reset_input_buffer()
        self.enviar_comando("eletroima", 1 if ligar else 0)
        return self._aguardar_ok(timeout or self.COMANDO_TIMEOUT, "eletroima")

    def obter_status(self, timeout=None):
        """Solicita e retorna o status atual do robô."""
        self.ser.reset_input_buffer()
        self.enviar_comando("status")
        msg = self.receber_mensagem(timeout=timeout or self.COMANDO_TIMEOUT)
        if msg is None:
            raise TimeoutError("ESP32 não respondeu ao pedido de status.")
        return msg


if __name__ == "__main__":
    # Teste manual
    try:
        with ESP32Communication() as esp:
            print("1. Solicitando status...")
            status = esp.obter_status()
            print(f"Status atual: {status}")

            print("\n2. Executando Homing...")
            esp.homing()
            print("Homing concluído!")

            print("\n3. Movimentando com Eletroímã...")
            # Exemplo: vai para (100, 50) com ímã desligado, liga ímã e vai para (200, 50), desliga ímã
            esp.gantry(
                x=[100.0, 200.0, 200.0],
                y=[50.0, 50.0, 50.0],
                eletroima=[0, 1, 0]
            )
            print("Movimento concluído com sucesso!")
    except KeyboardInterrupt:
        print("\nInterrompido (Ctrl+C)")
    except Exception as e:
        print(f"Erro: {e}")