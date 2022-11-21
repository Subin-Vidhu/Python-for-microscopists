//@ts-check
"use strict";


async function renderPage() {

    let res = {};
    try {
        res = await fetch("/api/profile").then(r => r.json());
    } catch {
        console.log("Not logged in.")
        return;
    }
    if (res?.["success"] != true) return;



    document.querySelector('[data-displayname]').innerHTML = res["displayName"] || res["clientname"];
    document.querySelector('[data-client-id]').innerHTML = res["idclient"];
    document.querySelector('[data-client-name]').innerHTML = res["clientname"];

    //login icon changes, filter, title, action(clear cookie or signout route)
    let eSignin = document.querySelector('[data-icon-signin]');
    eSignin.setAttribute("title", "Sign Out");
    eSignin.setAttribute("style", "filter: hue-rotate(17deg);");
    //debugger;

    eSignin.setAttribute("onclick", "client_logout()");
    eSignin.setAttribute("href", "/signout");


    //debugger;
};//
renderPage();

function client_logout() {
    //debugger;
    document.cookie = "";
    

};//
