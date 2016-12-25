var GET = function(str, resolve) {
  var xhr = new XMLHttpRequest();
  xhr.open("GET", str, true);
  xhr.setRequestHeader("Content-Type", "application/json");
  xhr.onload = function() {
    if (xhr.readyState === 4 && xhr.status === 200) {
      resolve(xhr.responseText);
    }
  };
  xhr.send();
};

var url = "https://api.github.com/repos/maierfelix/maierfelix.github.io";

GET(url, function(res) {
  var data = null;
  try {
    data = JSON.parse(res);
  } catch (e) { return void 0; }
  var date = new Date(data.pushed_at);
  var month = date.getUTCMonth()+1;
  var year = date.getUTCFullYear();
  var str = date.toLocaleString("en-us", { month: "long" }) + " " + year;
  var result = "Last updated: " + str;
  updated.innerHTML = result;
});