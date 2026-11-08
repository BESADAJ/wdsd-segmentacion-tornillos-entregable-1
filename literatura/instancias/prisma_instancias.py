# Diagrama PRISMA 2020 de la revisión de segmentación de instancias (búsqueda C4, 2026-11-07).
# Se puede pegar como celda del notebook (sección 3.1). Requiere solo matplotlib.
import matplotlib.pyplot as plt

AZUL, ROJO, VERDE = '#dbe9f6', '#f6e0db', '#e3f1dc'
fig, ax = plt.subplots(figsize=(11, 10))
ax.axis('off'); ax.set_xlim(0, 1); ax.set_ylim(0, 1)

def caja(x, y, texto, color):
    ax.text(x, y, texto, ha='center', va='center', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.5', fc=color, ec='#555'))

def flecha(x0, y0, x1, y1):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle='->', color='#333'))

# Columna izquierda: bases de datos. Columna derecha: excluidos. Extremo derecho: otras fuentes.
filas = [0.92, 0.76, 0.60, 0.44, 0.28, 0.10]
caja(0.27, filas[0], 'Registros identificados: 271\nWoS 43 · Scopus 62 · IEEE Xplore 166', AZUL)
caja(0.62, filas[0], 'Duplicados eliminados: 52', ROJO)
caja(0.27, filas[1], 'Registros cribados\n(título y resumen): 219', AZUL)
caja(0.62, filas[1], 'Excluidos: 202\n(sin tornillos, sujetadores,\ndesmontaje o remanufactura)', ROJO)
caja(0.27, filas[2], 'Registros del dominio: 17', AZUL)
caja(0.62, filas[2], 'Excluidos por resumen: 2\n(fuera de dominio: vértebras,\nhuellas de edificios)', ROJO)
caja(0.27, filas[3], 'Reportes buscados\nen texto completo: 15', AZUL)
caja(0.62, filas[3], 'No recuperados: 4\n(sin acceso al texto completo)', ROJO)
caja(0.27, filas[4], 'Reportes evaluados\nen texto completo: 11', AZUL)
caja(0.62, filas[4], 'Excluidos: 4\n(un solo modelo, sin comparar\nmodelos de instancias)', ROJO)
caja(0.27, filas[5], 'Incluidos en la síntesis: 9\n(7 de la búsqueda + 2 de otras fuentes)', VERDE)
caja(0.89, filas[0], 'Otras fuentes: 2\nKalitsios 2022\n(artículo de WDSD)\nChao 2025 (usa WDSD)', AZUL)

for i in range(len(filas) - 1):
    flecha(0.27, filas[i] - 0.045, 0.27, filas[i + 1] + 0.045)
    flecha(0.43, filas[i], 0.49, filas[i])
ax.plot([0.89, 0.89], [filas[0] - 0.07, filas[5]], color='#333', lw=1)
flecha(0.89, filas[5], 0.47, filas[5])
for y, t in zip([0.92, 0.68, 0.36, 0.10], ['Identificación', 'Cribado', 'Elegibilidad', 'Inclusión']):
    ax.text(0.01, y, t, rotation=90, va='center', fontsize=10, weight='bold', color='#335')
ax.set_title('Diagrama PRISMA 2020: segmentación de instancias (búsqueda C4, 2026-11-07)', fontsize=11)
plt.tight_layout()
plt.savefig('prisma_instancias.png', dpi=200, bbox_inches='tight')
plt.show()
