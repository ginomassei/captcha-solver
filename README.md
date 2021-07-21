# Captcha Solver API

<p>Esta API implementa una red neuronal convolucional desarrollada con pytorch para resolver captchas que tengan dígitos alfanuméricos. El método de resolución
que utiliza consiste en separar cada dígito de la imagen, y que el modelo resuelva de a un dígito, combinando los resultados en la salida
</p>

<p>La imagen es tratada con filtros, primero se lleva la imagen a escalas de grises para posteriormente hacer que estos valores tiendan
a 0 o 1 dependiendo de que tan cerca de un valor umbral estén situados. Luego se apliga un filtro de blur con un tamaño
de kernel de 3 x 3. De esta manera logro eliminar la mayoría de las líneas rectas de fondo.</p>

<h3>Modo de uso:</h3>
<p>Para construir la imagen <b>"docker build -t captcha-nn ."</b></p>
<p>Para iniciar el contenedor de docker <b>"docker run -p 5000:5000 --name nn-API captcha-nn"</b></p>

<p>Esta API tiene un endpont (GET) de testeo: <b>http://localhost/ping</b> que devuelve "Hello" si todo esta funcionando correctamente.</p>
