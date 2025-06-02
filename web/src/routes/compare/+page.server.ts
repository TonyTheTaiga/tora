import { error } from "@sveltejs/kit";
import type { PageServerLoad } from "./$types";

//export const load: PageServerLoad = async ({ url }) => {
//  const idsParam = url.searchParams.get("ids");
//
//  if (!idsParam) {
//    throw error(400, "Missing 'ids' query parameter");
//  }
//
//  const ids = idsParam.split(",").filter(Boolean); // Filter handles accidental trailing commas
//  console.log(ids);
//  return {
//    ids,
//  };
//};
