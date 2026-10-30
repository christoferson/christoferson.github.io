document.querySelectorAll('.snippet pre').forEach(function (pre) {
	var box = document.createElement('div');
	box.className = 'code';
	pre.parentNode.insertBefore(box, pre);
	box.appendChild(pre);
	var btn = document.createElement('button');
	btn.type = 'button';
	btn.className = 'copy';
	btn.textContent = 'Copy';
	btn.addEventListener('click', function () {
		navigator.clipboard.writeText(pre.innerText).then(function () {
			btn.textContent = 'Copied';
			btn.classList.add('done');
			setTimeout(function () { btn.textContent = 'Copy'; btn.classList.remove('done'); }, 1500);
		});
	});
	box.appendChild(btn);
});
