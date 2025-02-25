javascript:(function(){ 
    var address = window.location.href.split("/").pop(); 
    if (!/^[1-9A-HJ-NP-Za-km-z]{44}$/.test(address)) { 
        address = prompt("Enter CA:");
    }
    if (address) { 
        window.open("https://rugcheck.xyz/tokens/" + encodeURIComponent(address), "_blank");
    }
})();